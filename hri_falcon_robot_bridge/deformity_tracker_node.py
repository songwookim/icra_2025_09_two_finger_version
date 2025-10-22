#!/usr/bin/env python3
"""
RealSense 컬러 스트림 + HSV 마스킹 기반 변형(공/타원) 추적 노드
- 최대 컨투어에 대해 원/타원 특성 추출
- 반지름(또는 근사 타원 장/단축)과 circularity, eccentricity 산출
- GUI (옵션) 시각화
- 변형 지표(circularity / eccentricity)를 다른 노드들이 동기화 로깅 가능하도록 토픽 퍼블리시

ROS2 파라미터
- device_index (int, 기본 -1): 연결된 카메라 인덱스 선택(0,1,...)  -1이면 자동
- serial_number (str): 특정 시리얼 카메라 선택(우선순위 높음)
- width/height/fps (int): 컬러 스트림 해상도/프레임레이트
- enable_cv_window (bool): GUI 창 사용 여부 (기본 True, DISPLAY 있으면 자동)
"""
from __future__ import annotations

import os
import time
import csv
import math
from typing import Optional, Tuple

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool

try:
    import pyrealsense2 as rs  # type: ignore
except Exception as e:  # pragma: no cover
    rs = None  # type: ignore
    raise

try:
    import imutils  # type: ignore
    HAS_IMUTILS = True
except Exception:
    imutils = None  # type: ignore
    HAS_IMUTILS = False

# GUI 환경 설정
_ENABLE_CV_ENV = os.environ.get("ENABLE_CV_WINDOW", "").lower() in ("1","true","yes","on")
_HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
_WANT_GUI = _ENABLE_CV_ENV or _HAS_DISPLAY
if not _WANT_GUI:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
else:
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        try:
            del os.environ["QT_QPA_PLATFORM"]
        except Exception:
            pass

# HSV 색범위 프리셋 (기존과 동일)
COLOR_RANGES = {
    'yellow': [(np.array([20, 100, 100], np.uint8), np.array([35, 255, 255], np.uint8))],
    'green':  [(np.array([40, 60, 60],  np.uint8), np.array([85, 255, 255], np.uint8))],
    'blue':   [(np.array([90, 100, 60],  np.uint8), np.array([130, 255, 255], np.uint8))],
    'red':    [
        (np.array([0, 120, 70],  np.uint8), np.array([10, 255, 255], np.uint8)),
        (np.array([170, 120, 70], np.uint8), np.array([179, 255, 255], np.uint8)),
    ],
}
ACTIVE_COLORS = {
    'red',
    'green',
    'blue',
    'yellow',
}

class DeformityTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__('deformity_tracker_node')
        # 파라미터
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('device_index', -1)
        self.declare_parameter('serial_number', '')
        self.declare_parameter('enable_cv_window', True)
        self.declare_parameter('auto_fallback', True)
        self.declare_parameter('exclude_serials', '')
        self.declare_parameter('open_retry_count', 1)
        self.declare_parameter('open_retry_delay_sec', 0.4)
        # 캡처 저장 옵션 (로거 활성 시 캡처 자동 저장)
        self.declare_parameter('record_video_auto', True)
        self.declare_parameter('video_dir', '')
        self.declare_parameter('capture_mode', 'frames')  # 'frames' | 'video'
        self.declare_parameter('frame_save_interval', 5)
        self.declare_parameter('video_fourcc', 'XVID')
        self.declare_parameter('video_fps', 0.0)  # 0이면 camera fps 사용

        self.width = int(self.get_parameter('width').get_parameter_value().integer_value)
        self.height = int(self.get_parameter('height').get_parameter_value().integer_value)
        self.fps = int(self.get_parameter('fps').get_parameter_value().integer_value)
        self.device_index = int(self.get_parameter('device_index').get_parameter_value().integer_value)
        self.serial_number = str(self.get_parameter('serial_number').get_parameter_value().string_value)
        self.enable_cv_window = bool(self.get_parameter('enable_cv_window').get_parameter_value().bool_value)
        self.auto_fallback = bool(self.get_parameter('auto_fallback').get_parameter_value().bool_value)
        self.exclude_serials_raw = str(self.get_parameter('exclude_serials').get_parameter_value().string_value)
        self.open_retry_count = int(self.get_parameter('open_retry_count').get_parameter_value().integer_value)
        self.open_retry_delay_sec = float(self.get_parameter('open_retry_delay_sec').get_parameter_value().double_value)
        self.exclude_serials = set(s.strip() for s in self.exclude_serials_raw.split(',') if s.strip())
        if os.environ.get('RS_EXCLUDE_SERIALS'):
            for s in os.environ.get('RS_EXCLUDE_SERIALS', '').split(','):  # pragma: no branch
                if s.strip():
                    self.exclude_serials.add(s.strip())

        if _WANT_GUI and not self.enable_cv_window:
            self.enable_cv_window = True
            self.get_logger().info("GUI 환경 감지됨: CV 창 자동 활성화")

        # RealSense 파이프라인
        self.pipe = None
        self.active_serial = None
        try:
            self.pipe = self._open_realsense(self.serial_number, self.device_index)
        except Exception as e:
            self.get_logger().error(f"RealSense 초기화 실패: {e}")
            raise

        # CSV 로깅(기존 기능 유지)
        self._csv_enabled = False
        self._csv_f = None
        self._csv_writer = None
        # 이미지 캡처 상태
        self.record_video_auto = bool(self.get_parameter('record_video_auto').get_parameter_value().bool_value)
        self.video_dir = str(self.get_parameter('video_dir').get_parameter_value().string_value)
        self.capture_mode = str(self.get_parameter('capture_mode').get_parameter_value().string_value).strip().lower() or 'frames'
        if self.capture_mode not in {'frames','video'}:
            self.get_logger().warn(f"[CAPTURE] unsupported mode '{self.capture_mode}', falling back to 'frames'")
            self.capture_mode = 'frames'
        try:
            self.frame_save_interval = int(self.get_parameter('frame_save_interval').get_parameter_value().integer_value)
        except Exception:
            self.frame_save_interval = 5
        if self.frame_save_interval <= 0:
            self.frame_save_interval = 1
        self.video_fourcc = str(self.get_parameter('video_fourcc').get_parameter_value().string_value).strip() or 'XVID'
        try:
            self.video_fps = float(self.get_parameter('video_fps').get_parameter_value().double_value)
        except Exception:
            self.video_fps = 0.0
        self._recording = False
        self._record_dir = None
        self._frame_counter = 0
        self._video_writer = None

        # FPS 측정
        self._fps_t = time.time()
        self._fps_c = 0
        self._fps_val = 0.0

        self._cv_inited = False

        # Publishers (circularity + eccentricity)
        self.pub_circularity = self.create_publisher(Float32, '/deformity_tracker/circularity', 10)
        self.pub_eccentricity = self.create_publisher(Float32, '/deformity_tracker/eccentricity', 10)
        # Follow data_logger logging state to control video recording automatically
        self._logger_active = False
        try:
            self.create_subscription(Bool, '/data_logger/logging_active', self._on_logger_state, 10)
            self.get_logger().info('Subscribed /data_logger/logging_active (auto capture control)')
        except Exception as e:
            self.get_logger().warn(f'Logger state subscribe failed: {e}')

        self.get_logger().info("Deformity tracker node started")
        if self.record_video_auto:
            self.get_logger().info('record_video_auto=True → start/stop follows data_logger (press s to toggle logger)')

    # ---------------- RealSense helpers ----------------
    def _list_devices(self):
        ctx = rs.context()  # type: ignore[attr-defined]
        return [d for d in ctx.devices]

    def _open_realsense(self, serial: str, index: int):
        devs = self._list_devices()
        all_serials = [d.get_info(rs.camera_info.serial_number) for d in devs]  # type: ignore[attr-defined]
        if not all_serials:
            raise RuntimeError("No RealSense devices found")
        candidates = []
        if serial and serial in all_serials:
            candidates.append(serial)
        elif serial and serial not in all_serials:
            self.get_logger().warn(f"요청한 serial {serial} 을(를) 찾지 못함. fallback 진행")
        if not candidates and index >= 0 and index < len(all_serials):
            candidates.append(all_serials[index])
        for s in all_serials:
            if s not in candidates:
                candidates.append(s)
        if self.exclude_serials:
            before = list(candidates)
            candidates = [s for s in candidates if s not in self.exclude_serials]
            removed = [s for s in before if s not in candidates]
            if removed:
                self.get_logger().info(f"Exclude serials 제거됨: {removed}")

        self.get_logger().info(f"[RealSense] 후보 시리얼: {candidates}")
        last_err = None
        for cand in candidates:
            for attempt in range(self.open_retry_count + 1):
                pipe = rs.pipeline()  # type: ignore[attr-defined]
                cfg = rs.config()  # type: ignore[attr-defined]
                try:
                    cfg.enable_device(cand)
                    cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)  # type: ignore[attr-defined]
                    pipe.start(cfg)
                    self.active_serial = cand
                    self.get_logger().info(f"[RealSense] using serial={cand} (attempt {attempt+1})")
                    return pipe
                except Exception as e:
                    msg = str(e)
                    busy = ("Device or resource busy" in msg) or ("busy" in msg.lower())
                    last_err = e
                    self.get_logger().warn(f"[RealSense] open 실패(serial={cand}, attempt={attempt+1}): {e}")
                    try:
                        pipe.stop()
                    except Exception:
                        pass
                    if busy and attempt < self.open_retry_count:
                        time.sleep(self.open_retry_delay_sec)
                        continue
                    if busy and self.auto_fallback:
                        self.get_logger().info(f"[RealSense] serial {cand} busy → 다음 후보")
                        break
                    raise
        raise RuntimeError(f"모든 RealSense open 실패 (candidates={candidates}) 마지막 오류: {last_err}")

    # ---------------- Image processing ----------------
    def _extract_mask_and_contours(self, frame_bgr: np.ndarray):
        blurred = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        masks_by_color = {}
        for name, ranges in COLOR_RANGES.items():
            acc = None
            for lo, hi in ranges:
                mr = cv2.inRange(hsv, lo, hi)
                acc = mr if acc is None else cv2.bitwise_or(acc, mr)
            if acc is not None:
                masks_by_color[name] = acc
                mask_total = cv2.bitwise_or(mask_total, acc)
        kernel = np.ones((5, 5), np.uint8)
        mask_total = cv2.erode(mask_total, kernel, iterations=2)
        mask_total = cv2.dilate(mask_total, kernel, iterations=2)
        contours = cv2.findContours(mask_total.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if HAS_IMUTILS:
            cnts = imutils.grab_contours(contours)  # type: ignore[attr-defined]
        else:
            cnts = contours[0] if len(contours) == 2 else contours[1]
        return mask_total, masks_by_color, cnts

    def _measure_contour(self, cnt: np.ndarray):
        area = float(cv2.contourArea(cnt))
        if area <= 1e-3:
            return None
        peri = float(cv2.arcLength(cnt, True))
        circularity = 0.0
        if peri > 1e-6:
            circularity = float(4.0 * math.pi * area / (peri * peri))
        center = None
        ((x, y), r) = cv2.minEnclosingCircle(cnt)
        if r > 0:
            M = cv2.moments(cnt)
            if M.get("m00", 0) > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # Ellipse fit for eccentricity
        eccentricity = 0.0
        aspect = 0.0
        a = b = 0.0
        if len(cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(cnt)
                (ecx, ecy), (d1, d2), angle = ellipse
                a_raw, b_raw = d1 / 2.0, d2 / 2.0
                if a_raw < b_raw:
                    a_raw, b_raw = b_raw, a_raw
                a, b = a_raw, b_raw
                if a > 1e-6:
                    aspect = b / a
                    eccentricity = math.sqrt(max(0.0, 1.0 - (aspect * aspect)))
            except Exception:
                pass
        return {
            'center': center,
            'radius': float(r),
            'circularity': circularity,
            'eccentricity': eccentricity,
            'aspect': aspect,
            'a': a,
            'b': b,
            'area': area
        }

    # ---------------- Video helpers ----------------
    def _ensure_capture_dir(self):
        from pathlib import Path
        base_root = Path(self.video_dir) if self.video_dir else (Path.cwd() / 'outputs' / 'deformity_frames')
        base_root.mkdir(parents=True, exist_ok=True)
        return base_root

    def _start_capture(self):
        import datetime
        base = self._ensure_capture_dir()
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        from pathlib import Path
        path = base / f'{ts}_deformity'
        if self.capture_mode == 'video':
            try:
                fourcc = cv2.VideoWriter_fourcc(*self.video_fourcc)
                fps = self.video_fps if self.video_fps > 0 else float(self.fps)
                path.mkdir(parents=True, exist_ok=True)
                video_path = path / 'capture.avi'
                writer = cv2.VideoWriter(str(video_path), fourcc, fps, (int(self.width), int(self.height)))
                if not writer.isOpened():
                    raise RuntimeError(f'VideoWriter open failed (fourcc={self.video_fourcc})')
                self._video_writer = writer
                self._record_dir = path
                self._frame_counter = 0
                self._recording = True
                self.get_logger().info(f"[CAPTURE] video -> {video_path} (fps={fps:.1f}, fourcc={self.video_fourcc})")
            except Exception as e:
                self._video_writer = None
                self.capture_mode = 'frames'
                self.get_logger().warn(f"[CAPTURE] video mode failed ({e}); falling back to frames")
        if self.capture_mode == 'frames':
            path.mkdir(parents=True, exist_ok=True)
            self._record_dir = path
            self._frame_counter = 0
            self._recording = True
            self.get_logger().info(f"[CAPTURE] frames -> {path} (interval={self.frame_save_interval})")

    def _stop_capture(self):
        if self._recording and self._record_dir is not None:
            if self.capture_mode == 'video' and self._video_writer is not None:
                try:
                    self._video_writer.release()
                except Exception:
                    pass
                self._video_writer = None
                self.get_logger().info(f"[CAPTURE] saved video -> {self._record_dir / 'capture.avi'}")
            else:
                self.get_logger().info(f"[CAPTURE] saved frames -> {self._record_dir}")
        else:
            self.get_logger().info('[CAPTURE] stop (no active recording)')
        self._record_dir = None
        self._frame_counter = 0
        self._recording = False

    def _on_logger_state(self, msg: Bool) -> None:
        try:
            active = bool(msg.data)
        except Exception:
            active = False
        self._logger_active = active
        self.get_logger().info(f"[CAPTURE] logger active = {active}")
        if not self.record_video_auto:
            return
        if active and not self._recording:
            try:
                self._start_capture()
            except Exception as e:
                self.get_logger().warn(f"[CAPTURE] start failed: {e}")
        elif (not active) and self._recording:
            self._stop_capture()

    # ---------------- Main loop ----------------
    def spin(self):
        assert self.pipe is not None
        try:
            while rclpy.ok():
                frames = self.pipe.wait_for_frames()
                color = frames.get_color_frame()
                if not color:
                    continue
                frame = np.asanyarray(color.get_data())
                vis = frame.copy()

                mask, masks_by_color, cnts = self._extract_mask_and_contours(frame)
                best = None
                color_label = None
                if len(cnts) > 0:
                    c = max(cnts, key=cv2.contourArea)
                    best = self._measure_contour(c)
                    if best:
                        cmask = np.zeros(mask.shape, dtype=np.uint8)
                        cv2.drawContours(cmask, [c], -1, (255, 255, 255), -1)
                        best_name, best_overlap = None, -1
                        for name, m in masks_by_color.items():
                            overlap = int(cv2.countNonZero(cv2.bitwise_and(cmask, m)))
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_name = name
                        color_label = best_name
                        if best['center'] is not None and best['radius'] >= 10.0:
                            cv2.circle(vis, best['center'], 5, (0, 0, 255), -1)
                        # If ellipse data (a,b) present, draw approximate ellipse outline (use minEnclosingCircle already for circle) - optional
                        # Could extend: cv2.ellipse(...)
                circ_val = best['circularity'] if best else 0.0
                ecc_val = best['eccentricity'] if best else 0.0
                try:
                    self.pub_circularity.publish(Float32(data=float(circ_val)))
                    self.pub_eccentricity.publish(Float32(data=float(ecc_val)))
                except Exception:
                    pass
                # 이미지/비디오 캡처 저장
                if self._recording and self._record_dir is not None:
                    self._frame_counter += 1
                    if self.capture_mode == 'video':
                        if self._video_writer is not None:
                            try:
                                self._video_writer.write(frame)
                            except Exception:
                                pass
                    else:
                        if self._frame_counter % self.frame_save_interval == 0:
                            try:
                                fname = self._record_dir / f"frame_{self._frame_counter:06d}.png"
                                ok = cv2.imwrite(str(fname), frame)
                                if not ok:
                                    raise RuntimeError('cv2.imwrite returned False')
                            except Exception as e:
                                self.get_logger().warn(f"[CAPTURE] frame save failed: {e}")

                self._update_fps()
                prefix = f"[{color_label}] " if color_label else ""
                info = f"{prefix}circ:{circ_val:.3f} ecc:{ecc_val:.3f} FPS:{self._fps_val:.1f}"
                cv2.putText(vis, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(vis, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                # 녹화 상태 오버레이 표시
                try:
                    if self._recording:
                        rec_txt = 'REC'
                        color = (0, 0, 255) if self.capture_mode == 'video' else (0, 200, 255)
                        cv2.putText(vis, rec_txt, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(vis, rec_txt, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
                except Exception:
                    pass

                key = 255
                if self.enable_cv_window:
                    if not self._cv_inited:
                        try:
                            cv2.namedWindow('Deformity Tracking (RealSense Color)', cv2.WINDOW_NORMAL)
                            cv2.resizeWindow('Deformity Tracking (RealSense Color)', 960, 720)
                            cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
                            cv2.resizeWindow('Mask', 480, 360)
                            self._cv_inited = True
                        except Exception as e:
                            self.get_logger().warn(f"OpenCV 창 생성 실패: {e}")
                            self.enable_cv_window = False
                    if self.enable_cv_window:
                        cv2.imshow('Deformity Tracking (RealSense Color)', vis)
                        cv2.imshow('Mask', mask)
                        key = cv2.waitKey(1) & 0xFF

                # CSV 로깅 (기존 유지)
                if self._csv_enabled and self._csv_writer is not None:
                    t = time.time()
                    if best and best['center'] is not None and best['radius'] >= 10.0:
                        self._csv_writer.writerow({'t': f"{t:.3f}", 'cx': best['center'][0], 'cy': best['center'][1], 'r_px': int(best['radius']), 'circ': f"{circ_val:.3f}", 'ecc': f"{ecc_val:.3f}"})
                    else:
                        self._csv_writer.writerow({'t': f"{t:.3f}", 'cx': '', 'cy': '', 'r_px': '', 'circ': '', 'ecc': ''})

                if key != 255:
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        if not self._csv_enabled:
                            try:
                                self._csv_f = open('deformity_track_log.csv', 'w', newline='')
                                self._csv_writer = csv.DictWriter(self._csv_f, fieldnames=['t','cx','cy','r_px','circ','ecc'])
                                self._csv_writer.writeheader()
                                self._csv_enabled = True
                                self.get_logger().info('[CSV] deformity_track_log.csv START')
                            except Exception as e:
                                self.get_logger().warn(f'[CSV] start failed: {e}')
                        else:
                            try:
                                if self._csv_f:
                                    self._csv_f.close()
                                self._csv_f = None
                                self._csv_writer = None
                                self._csv_enabled = False
                                self.get_logger().info('[CSV] STOP & saved')
                            except Exception:
                                pass
                    elif key == ord('v'):
                        # 캡처 토글 (record_video_auto=True면 logger 활성 구간에서만 허용)
                        if self.record_video_auto and not self._logger_active:
                            self.get_logger().info('[CAPTURE] logger inactive; ignoring manual start (record_video_auto=True)')
                        else:
                            if not self._recording:
                                try:
                                    self._start_capture()
                                except Exception as e:
                                    self.get_logger().warn(f"[CAPTURE] start failed: {e}")
                            else:
                                self._stop_capture()
                    elif key == ord('p'):
                        # 현재 프레임 스냅샷 저장
                        try:
                            base = self._ensure_capture_dir()
                            import datetime
                            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                            img_path = os.path.join(str(base), f'{ts}_snap.png')
                            cv2.imwrite(img_path, frame)
                            self.get_logger().info(f"[SNAP] saved -> {img_path}")
                        except Exception as e:
                            self.get_logger().warn(f"[SNAP] save failed: {e}")
        finally:
            try:
                if self.pipe is not None:
                    self.pipe.stop()
            except Exception:
                pass
            # 캡처 자원 정리
            try:
                self._stop_capture()
            except Exception:
                pass
            try:
                if self.enable_cv_window:
                    cv2.destroyAllWindows()
            except Exception:
                pass

    def _update_fps(self):
        self._fps_c += 1
        now = time.time()
        if now - self._fps_t >= 1.0:
            self._fps_val = self._fps_c / (now - self._fps_t)
            self._fps_t = now
            self._fps_c = 0


def main(args=None):
    rclpy.init(args=args)
    node = DeformityTrackerNode()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
