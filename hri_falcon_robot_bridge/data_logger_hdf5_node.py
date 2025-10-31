#!/usr/bin/env python3
"""ROS 2 data logger that streams synchronized demos to CSV and robomimic-style HDF5."""
from __future__ import annotations

import csv
import datetime as dt
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py  # type: ignore
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, String


class DataLoggerHDF5Node(Node):
    def __init__(self) -> None:
        super().__init__("data_logger_hdf5_node")
        self.declare_parameter("rate_hz", 100.0)
        self.declare_parameter("output_dir", "")
        self.declare_parameter("output_format", "both")  # csv | hdf5 | both
        self.declare_parameter("hdf5_chunk_size", 200)
        self.declare_parameter("episode_timeout_sec", 5.0)
        self.declare_parameter("start_immediately", False)
        self.declare_parameter("action_topic", "/impedance/stiffness_command")
        self.declare_parameter("action_dim", 3)
        self.declare_parameter("obs_modalities", ["ee_pos", "force", "deform", "emg"])

        self.rate_hz = float(self.get_parameter("rate_hz").get_parameter_value().double_value)
        output_dir_param = str(self.get_parameter("output_dir").get_parameter_value().string_value)
        self.output_dir = Path(output_dir_param) if output_dir_param else self._default_output_dir()
        self.output_format = (
            str(self.get_parameter("output_format").get_parameter_value().string_value)
            .strip()
            .lower()
        )
        if self.output_format not in {"csv", "hdf5", "both"}:
            self.output_format = "both"
        self.chunk_size = max(1, int(self.get_parameter("hdf5_chunk_size").get_parameter_value().integer_value))
        self.episode_timeout = float(
            self.get_parameter("episode_timeout_sec").get_parameter_value().double_value
        )
        self._logging_active = bool(
            self.get_parameter("start_immediately").get_parameter_value().bool_value
        )
        self.action_topic = str(self.get_parameter("action_topic").get_parameter_value().string_value)
        self.action_dim = max(
            1, int(self.get_parameter("action_dim").get_parameter_value().integer_value)
        )
        self.obs_modalities = list(
            self.get_parameter("obs_modalities").get_parameter_value().string_array_value
        )

        self._buffer: deque[Dict[str, Any]] = deque(maxlen=self.chunk_size)
        self._episode: List[Dict[str, Any]] = []
        self._episode_count = 0
        self._last_tick_ts: Optional[float] = None

        self._hdf5_file: Optional[h5py.File] = None
        self._csv_file: Optional[Any] = None
        self._csv_writer: Optional[Any] = None

        self._force: Dict[int, Optional[np.ndarray]] = {2: None, 3: None}
        self._ee_pose: Optional[np.ndarray] = None
        self._deform_circ: Optional[float] = None
        self._deform_ecc: Optional[float] = None
        self._emg: Optional[np.ndarray] = None
        self._action: Optional[np.ndarray] = None

        self._setup_subscriptions()

        period = max(0.001, 1.0 / max(1e-6, self.rate_hz))
        self.timer = self.create_timer(period, self._on_timer)

        if self._logging_active:
            self._open_outputs()
        self.get_logger().info(
            f"DataLoggerHDF5 ready (rate={self.rate_hz:.1f}Hz, format={self.output_format})"
        )

    # ----- setup -----
    def _setup_subscriptions(self) -> None:
        self.create_subscription(WrenchStamped, "/force_sensor/s2/wrench", self._on_force_s2, 20)
        self.create_subscription(WrenchStamped, "/force_sensor/s3/wrench", self._on_force_s3, 20)
        self.create_subscription(PoseStamped, "/ee_pose", self._on_ee_pose, 20)
        self.create_subscription(Float32, "/deformity_tracker/circularity", self._on_deform_circ, 10)
        self.create_subscription(Float32, "/deformity_tracker/eccentricity", self._on_deform_ecc, 10)
        self.create_subscription(Float32MultiArray, "/emg/raw", self._on_emg, 20)
        self.create_subscription(Float32MultiArray, self.action_topic, self._on_action, 20)
        self.create_subscription(String, "/hand_tracker/key", self._on_key, 10)

    def _default_output_dir(self) -> Path:
        ws_root = Path(__file__).resolve().parents[4]
        stamp = dt.datetime.now().strftime("%Y%m%d")
        return ws_root / "outputs" / "robomimic_data" / stamp

    # ----- callbacks -----
    def _on_force_s2(self, msg: WrenchStamped) -> None:
        self._on_force(2, msg)

    def _on_force_s3(self, msg: WrenchStamped) -> None:
        self._on_force(3, msg)

    def _on_force(self, idx: int, msg: WrenchStamped) -> None:
        force = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ],
            dtype=np.float32,
        )
        self._force[idx] = force

    def _on_ee_pose(self, msg: PoseStamped) -> None:
        self._ee_pose = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float32,
        )

    def _on_deform_circ(self, msg: Float32) -> None:
        self._deform_circ = float(msg.data)

    def _on_deform_ecc(self, msg: Float32) -> None:
        self._deform_ecc = float(msg.data)

    def _on_emg(self, msg: Float32MultiArray) -> None:
        self._emg = np.asarray(msg.data[:8], dtype=np.float32)

    def _on_action(self, msg: Float32MultiArray) -> None:
        data = np.asarray(msg.data[: self.action_dim], dtype=np.float32)
        if data.shape[0] < self.action_dim:
            pad = np.zeros(self.action_dim, dtype=np.float32)
            pad[: data.shape[0]] = data
            data = pad
        self._action = data

    def _on_key(self, msg: String) -> None:
        key = str(msg.data).strip().lower()
        if key == "s":
            if not self._logging_active:
                self._logging_active = True
                self._open_outputs()
                self.get_logger().info("Logging started")
            else:
                self._save_episode()
                self._close_outputs()
                self._logging_active = False
                self.get_logger().info("Logging stopped")
        elif key == "e" and self._logging_active:
            self._save_episode()
            self.get_logger().info("Episode boundary")

    # ----- IO helpers -----
    def _open_outputs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.output_format in {"hdf5", "both"}:
            path = self.output_dir / f"{stamp}_demo.hdf5"
            self._hdf5_file = h5py.File(path, "w")
            self._init_hdf5_metadata(self._hdf5_file)
            self.get_logger().info(f"HDF5 -> {path}")
        if self.output_format in {"csv", "both"}:
            path = self.output_dir / f"{stamp}_demo.csv"
            self._csv_file = open(path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._write_csv_header(self._csv_writer)
            self.get_logger().info(f"CSV -> {path}")
        self._buffer.clear()
        self._episode.clear()
        self._last_tick_ts = None

    def _close_outputs(self) -> None:
        if self._hdf5_file is not None:
            self._hdf5_file.flush()
            self._hdf5_file.close()
            self._hdf5_file = None
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        self._buffer.clear()
        self._episode.clear()
        self._last_tick_ts = None

    def _init_hdf5_metadata(self, handle: h5py.File) -> None:
        handle.attrs["created"] = dt.datetime.now().isoformat()
        handle.attrs["rate_hz"] = self.rate_hz
        handle.attrs["action_dim"] = self.action_dim
        env_meta = {
            "env_name": "TwoFingerPuncture",
            "type": 1,
            "env_kwargs": {"action_dim": self.action_dim, "obs_keys": self.obs_modalities},
        }
        handle.attrs["env_args"] = json.dumps(env_meta)
        handle.create_group("data")

    def _write_csv_header(self, writer: Any) -> None:
        header = ["timestamp", "episode"]
        header += [f"s2_{axis}" for axis in ("fx", "fy", "fz", "tx", "ty", "tz")]
        header += [f"s3_{axis}" for axis in ("fx", "fy", "fz", "tx", "ty", "tz")]
        header += ["ee_px", "ee_py", "ee_pz", "ee_qx", "ee_qy", "ee_qz", "ee_qw"]
        header += ["deform_circ", "deform_ecc"]
        header += [f"emg_ch{i}" for i in range(1, 9)]
        header += [f"action_{i}" for i in range(self.action_dim)]
        writer.writerow(header)

    # ----- logging loop -----
    def _on_timer(self) -> None:
        if not self._logging_active:
            return
        obs = self._build_observation()
        if obs is None:
            return
        now_sec = float(self.get_clock().now().nanoseconds) / 1e9
        sample = {
            "timestamp": now_sec,
            "obs": obs,
            "action": self._action.copy() if self._action is not None else np.zeros(self.action_dim, dtype=np.float32),
        }
        self._buffer.append(sample)
        self._episode.append(sample)
        self._write_csv_samples()
        self._flush_episode_if_idle(now_sec)

    def _build_observation(self) -> Optional[Dict[str, np.ndarray]]:
        if self._ee_pose is None:
            return None
        obs: Dict[str, np.ndarray] = {}
        if "ee_pos" in self.obs_modalities:
            obs["ee_pos"] = self._ee_pose[:3]
        if "ee_quat" in self.obs_modalities:
            obs["ee_quat"] = self._ee_pose[3:]
        if "force" in self.obs_modalities:
            if self._force[2] is not None and self._force[3] is not None:
                obs["force"] = (self._force[2][:3] + self._force[3][:3]) / 2.0
            elif self._force[2] is not None:
                obs["force"] = self._force[2][:3]
            elif self._force[3] is not None:
                obs["force"] = self._force[3][:3]
            else:
                obs["force"] = np.zeros(3, dtype=np.float32)
        if "deform" in self.obs_modalities:
            circ = self._deform_circ if self._deform_circ is not None else 0.0
            ecc = self._deform_ecc if self._deform_ecc is not None else 0.0
            obs["deform"] = np.asarray([circ, ecc], dtype=np.float32)
        if "emg" in self.obs_modalities:
            obs["emg"] = self._emg if self._emg is not None else np.zeros(8, dtype=np.float32)
        return obs

    def _write_csv_samples(self) -> None:
        if self._csv_writer is None:
            return
        while self._buffer:
            sample = self._buffer.popleft()
            row: List[Any] = [sample["timestamp"], self._episode_count]
            s2 = self._force[2] if self._force[2] is not None else np.zeros(6, dtype=np.float32)
            s3 = self._force[3] if self._force[3] is not None else np.zeros(6, dtype=np.float32)
            row += s2.tolist()
            row += s3.tolist()
            row += (self._ee_pose if self._ee_pose is not None else np.zeros(7, dtype=np.float32)).tolist()
            circ = self._deform_circ if self._deform_circ is not None else 0.0
            ecc = self._deform_ecc if self._deform_ecc is not None else 0.0
            row += [circ, ecc]
            row += (self._emg if self._emg is not None else np.zeros(8, dtype=np.float32)).tolist()
            row += sample["action"].tolist()
            self._csv_writer.writerow(row)
        if self._csv_file is not None:
            self._csv_file.flush()

    def _flush_episode_if_idle(self, now_sec: float) -> None:
        if self._last_tick_ts is None:
            self._last_tick_ts = now_sec
            return
        if (now_sec - self._last_tick_ts) >= self.episode_timeout:
            self._save_episode()
            self._last_tick_ts = None
        else:
            self._last_tick_ts = now_sec

    def _save_episode(self) -> None:
        if not self._episode:
            return
        if self._hdf5_file is not None:
            group = self._hdf5_file["data"].create_group(f"demo_{self._episode_count}")
            obs_group = group.create_group("obs")
            keys = self._episode[0]["obs"].keys()
            for key in keys:
                data = np.stack([step["obs"][key] for step in self._episode], axis=0)
                obs_group.create_dataset(key, data=data, compression="gzip")
            actions = np.stack([step["action"] for step in self._episode], axis=0)
            group.create_dataset("actions", data=actions, compression="gzip")
            rewards = np.zeros(actions.shape[0], dtype=np.float32)
            group.create_dataset("rewards", data=rewards)
            dones = np.zeros(actions.shape[0], dtype=np.bool_)
            dones[-1] = True
            group.create_dataset("dones", data=dones)
            duration = self._episode[-1]["timestamp"] - self._episode[0]["timestamp"]
            group.attrs["num_samples"] = actions.shape[0]
            group.attrs["duration"] = float(duration)
            self._hdf5_file.flush()
        self._episode.clear()
        self._buffer.clear()
        self._episode_count += 1

    # ----- shutdown -----
    def destroy_node(self) -> None:  # type: ignore[override]
        self._save_episode()
        self._close_outputs()
        super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = DataLoggerHDF5Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()