#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / 'src'
PKG_SRC_ROOT = SRC_ROOT / 'hri_falcon_robot_bridge'
for candidate in (SRC_ROOT, PKG_SRC_ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from omegaconf import OmegaConf
except Exception as exc:  # pragma: no cover - requires optional dependency
    print('OmegaConf is required for this script: pip install omegaconf', file=sys.stderr)
    raise

from dynamixel_sdk import COMM_SUCCESS
from hri_falcon_robot_bridge.dynamixel_control import DynamixelControl

VELOCITY_UNIT_RAD_S = 0.229 * (2.0 * np.pi / 60.0)
CONFIG_PATH = WORKSPACE_ROOT / 'src' / 'hri_falcon_robot_bridge' / 'resource' / 'robot_parameter' / 'config.yaml'
DEFAULT_PROFILE_DIR = WORKSPACE_ROOT / 'outputs' / 'analysis' / 'stiffness_profiles'


@dataclass(frozen=True)
class ProfileData:
    time: np.ndarray
    stiffness: np.ndarray
    displacement: Optional[np.ndarray]
    reference: Optional[np.ndarray]


def _load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f'Missing robot config: {CONFIG_PATH}')
    return OmegaConf.load(str(CONFIG_PATH))


def _parse_vector(text: str | None, width: int, default: float) -> np.ndarray:
    if text is None:
        return np.full(width, default, dtype=float)
    raw = [v for v in text.split(',') if v.strip()]
    if not raw:
        return np.full(width, default, dtype=float)
    values = np.array([float(v) for v in raw], dtype=float)
    if values.size == 1:
        return np.full(width, values.item(), dtype=float)
    if values.size != width:
        raise ValueError(f'Expected {width} values, received {values.size}')
    return values


def _find_latest_profile(profile_dir: Path) -> Optional[Path]:
    profile_dir = profile_dir.expanduser()
    if not profile_dir.exists():
        return None
    candidates = list(profile_dir.glob('*_stiffness_profile.csv'))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _load_profile_csv(path: Path) -> ProfileData:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f'Profile CSV {path} contains no rows')

    if 'time_s' not in df.columns:
        raise KeyError("Profile CSV missing 'time_s' column")
    time = df['time_s'].to_numpy(dtype=float)

    stiffness_cols = ['Kx', 'Ky', 'Kz']
    if not set(stiffness_cols).issubset(df.columns):
        raise KeyError('Profile CSV missing stiffness columns Kx/Ky/Kz')
    stiffness = df[stiffness_cols].to_numpy(dtype=float)

    displacement = None
    disp_cols = ['disp_x', 'disp_y', 'disp_z']
    if set(disp_cols).issubset(df.columns):
        displacement = df[disp_cols].to_numpy(dtype=float)

    reference = None
    for col in df.columns:
        if col.endswith('_px') and len(col) > 3:
            prefix = col[:-3]
            cols = [f'{prefix}_px', f'{prefix}_py', f'{prefix}_pz']
            if all(c in df.columns for c in cols):
                reference = df[cols].to_numpy(dtype=float)
                break

    return ProfileData(time=time, stiffness=stiffness, displacement=displacement, reference=reference)


def _sample_profile(time: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    if time.size == 0:
        raise ValueError('Profile time vector is empty')
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if t <= time[0]:
        return values[0]
    if t >= time[-1]:
        return values[-1]
    out = np.empty(values.shape[1], dtype=float)
    for idx in range(values.shape[1]):
        out[idx] = float(np.interp(t, time, values[:, idx]))
    return out


def _write_currents(
    ctrl: DynamixelControl,
    currents: np.ndarray,
    limit: int,
    active_indices: Optional[List[int]] = None,
) -> None:
    addr = ctrl.cfg.control_table.ADDR_GOAL_CURRENT
    for idx, motor_id in enumerate(ctrl.cfg.ids):
        if active_indices is not None and idx not in active_indices:
            continue
        val = int(round(currents[idx]))
        if val > limit:
            val = limit
        elif val < -limit:
            val = -limit
        tx_val = val & 0xFFFF
        result, error = ctrl.packetHandler.write2ByteTxRx(ctrl.portHandler, motor_id, addr, tx_val)
        if result != COMM_SUCCESS:
            detail = ctrl.packetHandler.getTxRxResult(result)
            raise RuntimeError(f'write2ByteTxRx failed for ID {motor_id}: {detail}')
        if error != 0:
            detail = ctrl.packetHandler.getRxPacketError(error)
            raise RuntimeError(f'Dynamixel error for ID {motor_id}: {detail}')


def _restore_modes(ctrl: DynamixelControl, original: Dict[int, int]) -> None:
    ctrl.disable_torque()
    for motor_id, mode in original.items():
        ctrl.set_operating_mode_one(motor_id, mode)
    ctrl.enable_torque()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Simple impedance control loop for the DClaw gripper.')
    parser.add_argument('--kp', type=str, default='5.0', help='Joint stiffness gain(s) [N·m/rad].')
    parser.add_argument('--kd', type=str, default='0.2', help='Joint damping gain(s) [N·m·s/rad].')
    parser.add_argument('--current-scale', type=str, default='80.0', help='Current counts per commanded torque [cnt/N·m].')
    parser.add_argument('--current-limit', type=int, default=10, help='Saturation on current command magnitude (counts).')
    parser.add_argument('--target-offset', type=str, default=None, help='Desired joint offset(s) relative to measured pose [rad].')
    parser.add_argument('--duration', type=float, default=0.0, help='Stop after N seconds (0 keeps running).')
    parser.add_argument('--dt', type=float, default=0.01, help='Control period in seconds.')
    parser.add_argument('--velocity-alpha', type=float, default=0.3, help='EMA factor for velocity filtering [0-1].')
    parser.add_argument('--log', action='store_true', help='Print state each iteration.')
    parser.add_argument('--profile-csv', type=str, default='', help='Path to stiffness profile CSV (auto-detect latest if empty).')
    parser.add_argument('--profile-dir', type=str, default=str(DEFAULT_PROFILE_DIR), help='Directory searched for latest stiffness profile.')
    parser.add_argument('--reference-scale', type=float, default=8.0, help='Scale factor [rad/m] applied to displacement/reference deltas.')
    parser.add_argument('--profile-repeat', action='store_true', help='Repeat profile after it finishes.')
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = _load_config()
    dyn_cfg = getattr(cfg, 'dynamixel')
    controller = DynamixelControl(dyn_cfg)
    controller.connect()

    motor_ids = list(dyn_cfg.ids)
    width = len(motor_ids)

    # Split fingers: IDs in 20s -> finger 2, IDs in 30s -> finger 3
    f2_indices = [idx for idx, mid in enumerate(motor_ids) if 20 <= int(mid) < 30]
    f3_indices = [idx for idx, mid in enumerate(motor_ids) if 30 <= int(mid) < 40]
    if not f3_indices:
        raise RuntimeError('Unable to identify third finger IDs (expected IDs in the 30-range).')

    original_modes = {mid: controller.get_operating_mode(mid) for mid in motor_ids}
    force_mode = dyn_cfg.control_modes.current_control_mode
    controller.disable_torque()
    for idx in f3_indices:
        motor_id = motor_ids[idx]
        if original_modes[motor_id] != force_mode:
            controller.set_operating_mode_one(motor_id, force_mode)
    controller.enable_torque()

    kp = _parse_vector(args.kp, width, 5.0)
    kd = _parse_vector(args.kd, width, 0.2)
    current_scale = _parse_vector(args.current_scale, width, 80.0)
    offset = _parse_vector(args.target_offset, width, 0.0)
    current_limit = int(max(0, args.current_limit))

    target = controller.get_joint_positions(type='rad') + offset
    base_target = np.array(target, dtype=float)
    filt_velocity = np.zeros(width, dtype=float)

    profile: Optional[ProfileData] = None
    profile_path: Optional[Path]
    if args.profile_csv:
        profile_path = Path(args.profile_csv).expanduser()
        if not profile_path.exists():
            raise FileNotFoundError(f'Profile CSV not found: {profile_path}')
    else:
        profile_path = _find_latest_profile(Path(args.profile_dir))

    if profile_path:
        profile = _load_profile_csv(profile_path)
        print(f'Loaded stiffness profile: {profile_path}')
    else:
        profile_path = None
        print('No stiffness profile will be used (static gains).')

    running = True

    def _stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    start_time = time.monotonic()
    next_time = start_time

    try:
        while running:
            now = time.monotonic()
            if args.duration > 0.0 and (now - start_time) >= args.duration:
                break

            positions = controller.get_joint_positions(type='rad')
            velocities_raw = controller.get_joint_velocities()
            velocities = velocities_raw * VELOCITY_UNIT_RAD_S
            filt_velocity = args.velocity_alpha * velocities + (1.0 - args.velocity_alpha) * filt_velocity

            if profile is not None and profile.time.size:
                elapsed = now - start_time
                profile_end = float(profile.time[-1])
                profile_t = float(min(elapsed, profile_end))
                if args.profile_repeat and profile_end > 1e-6:
                    profile_t = float(elapsed % profile_end)

                k_sample = _sample_profile(profile.time, profile.stiffness, profile_t)
                k_vec = np.nan_to_num(np.asarray(k_sample, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                for axis, idx in enumerate(f3_indices):
                    if axis < k_vec.shape[0]:
                        kp[idx] = max(float(k_vec[axis]), 0.0)

                disp_sample: Optional[np.ndarray] = None
                if profile.displacement is not None:
                    disp_sample = _sample_profile(profile.time, profile.displacement, profile_t)
                elif profile.reference is not None:
                    ref_sample = _sample_profile(profile.time, profile.reference, profile_t)
                    ref0 = profile.reference[0]
                    disp_sample = ref_sample - ref0

                if disp_sample is not None:
                    disp_vec = np.nan_to_num(np.asarray(disp_sample, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                    for axis, idx in enumerate(f3_indices):
                        if axis < disp_vec.shape[0]:
                            target[idx] = base_target[idx] + float(disp_vec[axis]) * args.reference_scale

            error = target - positions
            torque_cmd = kp * error - kd * filt_velocity
            current_cmd = torque_cmd * current_scale
            current_cmd = np.clip(current_cmd, -current_limit, current_limit)

            _write_currents(controller, current_cmd, current_limit, active_indices=f3_indices)

            if args.log:
                print('err:', np.round(error, 4), 'vel:', np.round(filt_velocity, 4), 'cur:', np.round(current_cmd, 1))

            next_time += args.dt
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.monotonic()
    finally:
        try:
            _write_currents(controller, np.zeros(width, dtype=float), current_limit, active_indices=f3_indices)
        except Exception as exc:
            print(f'Failed to zero currents: {exc}', file=sys.stderr)
        try:
            _restore_modes(controller, original_modes)
        except Exception as exc:
            print(f'Failed to restore operating modes: {exc}', file=sys.stderr)
        controller.close_port()


if __name__ == '__main__':
    main()
