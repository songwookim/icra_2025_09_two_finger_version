#!/usr/bin/env python3
"""Convert synchronized CSV logs into robomimic-style demonstration HDF5 files."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import h5py  # type: ignore
import numpy as np


DEFAULT_OBS = ["ee_pos", "force", "deform", "emg"]
EMG_COLUMNS = [f"emg_ch{i}" for i in range(1, 9)]
FORCE_COMPONENTS = ("fx", "fy", "fz", "tx", "ty", "tz")
OBS_REQUIRED_COLUMNS: Dict[str, Set[str]] = {
    "ee_pos": {"ee_px", "ee_py", "ee_pz"},
    "ee_quat": {"ee_qx", "ee_qy", "ee_qz", "ee_qw"},
    "force": {f"s2_{comp}" for comp in FORCE_COMPONENTS} | {f"s3_{comp}" for comp in FORCE_COMPONENTS},
    "force_s2": {f"s2_{comp}" for comp in FORCE_COMPONENTS},
    "force_s3": {f"s3_{comp}" for comp in FORCE_COMPONENTS},
    "deform": {"deform_circ", "deform_ecc"},
    "emg": set(EMG_COLUMNS),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CSV produced by data_logger_node.py into robomimic-style HDF5 demos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv", type=Path, help="Input CSV file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HDF5 path (defaults to same stem as CSV)",
    )
    parser.add_argument(
        "--obs",
        nargs="+",
        default=DEFAULT_OBS,
        help="Observation keys to export (subset of: ee_pos, ee_quat, force, force_s2, force_s3, deform, emg)",
    )
    parser.add_argument(
        "--action-columns",
        nargs="+",
        default=None,
        help="CSV columns to use as actions; omit to fill zeros",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=3,
        help="Action dimensionality when action columns are not provided",
    )
    parser.add_argument(
        "--episode-column",
        type=str,
        default=None,
        help="Optional CSV column that encodes episode IDs",
    )
    parser.add_argument(
        "--episode-gap-sec",
        type=float,
        default=5.0,
        help="Start a new episode when the timestamp gap exceeds this value (ignored if --episode-column is used)",
    )
    parser.add_argument(
        "--no-gap-split",
        action="store_true",
        help="Disable gap-based episode splitting; useful when CSV already represents a single demo",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="TwoFingerPuncture",
        help="Env metadata stored in the HDF5 root attrs",
    )
    return parser.parse_args()


def safe_float(value: Optional[str], default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_observation(row: Dict[str, str], obs_keys: Sequence[str]) -> Dict[str, np.ndarray]:
    obs: Dict[str, np.ndarray] = {}
    if "ee_pos" in obs_keys:
        obs["ee_pos"] = np.asarray(
            [safe_float(row.get("ee_px")), safe_float(row.get("ee_py")), safe_float(row.get("ee_pz"))],
            dtype=np.float32,
        )
    if "ee_quat" in obs_keys:
        quat_cols = ("ee_qx", "ee_qy", "ee_qz", "ee_qw")
        values = [safe_float(row.get(col)) for col in quat_cols]
        obs["ee_quat"] = np.asarray(values, dtype=np.float32)
    if "force" in obs_keys or "force_s2" in obs_keys or "force_s3" in obs_keys:
        s2 = np.asarray([safe_float(row.get(f"s2_{comp}")) for comp in FORCE_COMPONENTS], dtype=np.float32)
        s3 = np.asarray([safe_float(row.get(f"s3_{comp}")) for comp in FORCE_COMPONENTS], dtype=np.float32)
        if "force" in obs_keys:
            obs["force"] = (s2[:3] + s3[:3]) / 2.0
        if "force_s2" in obs_keys:
            obs["force_s2"] = s2
        if "force_s3" in obs_keys:
            obs["force_s3"] = s3
    if "deform" in obs_keys:
        obs["deform"] = np.asarray(
            [safe_float(row.get("deform_circ")), safe_float(row.get("deform_ecc"))],
            dtype=np.float32,
        )
    if "emg" in obs_keys:
        obs["emg"] = np.asarray([safe_float(row.get(col)) for col in EMG_COLUMNS], dtype=np.float32)
    return obs


def compute_timestamp(row: Dict[str, str]) -> float:
    sec = safe_float(row.get("t_sec"), 0.0)
    nsec = safe_float(row.get("t_nanosec"), 0.0)
    return sec + (nsec * 1e-9)


def iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header row")
        for row in reader:
            yield row


def segment_episodes(
    rows: Iterable[Dict[str, str]],
    obs_keys: Sequence[str],
    action_columns: Optional[Sequence[str]],
    action_dim: int,
    episode_column: Optional[str],
    episode_gap_sec: float,
    no_gap_split: bool,
) -> Tuple[List[Tuple[List[Dict[str, np.ndarray]], List[np.ndarray], List[float]]], float]:
    episodes: List[Tuple[List[Dict[str, np.ndarray]], List[np.ndarray], List[float]]] = []
    all_dt: List[float] = []
    current_obs: List[Dict[str, np.ndarray]] = []
    current_actions: List[np.ndarray] = []
    current_ts: List[float] = []
    prev_ts: Optional[float] = None
    prev_episode_id: Optional[str] = None

    for row in rows:
        timestamp = compute_timestamp(row)
        obs = build_observation(row, obs_keys)
        action = build_action(row, action_columns, action_dim)
        episode_id = row.get(episode_column) if episode_column else None

        new_episode = False
        if episode_column:
            if prev_episode_id is None:
                prev_episode_id = episode_id
            elif episode_id != prev_episode_id:
                new_episode = True
                prev_episode_id = episode_id
        elif not no_gap_split and prev_ts is not None and (timestamp - prev_ts) > episode_gap_sec:
            new_episode = True

        if new_episode and current_obs:
            episodes.append((current_obs, current_actions, current_ts))
            current_obs, current_actions, current_ts = [], [], []

        current_obs.append(obs)
        current_actions.append(action)
        current_ts.append(timestamp)

        if prev_ts is not None:
            dt = timestamp - prev_ts
            if dt > 0.0:
                all_dt.append(dt)
        prev_ts = timestamp

    if current_obs:
        episodes.append((current_obs, current_actions, current_ts))

    median_dt = float(np.median(all_dt)) if all_dt else 0.0
    return episodes, median_dt


def build_action(row: Dict[str, str], action_columns: Optional[Sequence[str]], action_dim: int) -> np.ndarray:
    if action_columns:
        values = [safe_float(row.get(col)) for col in action_columns]
        return np.asarray(values, dtype=np.float32)
    return np.zeros(action_dim, dtype=np.float32)


def write_hdf5(
    output_path: Path,
    csv_path: Path,
    episodes: List[Tuple[List[Dict[str, np.ndarray]], List[np.ndarray], List[float]]],
    obs_keys: Sequence[str],
    action_dim: int,
    nominal_dt: float,
    env_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.attrs["created"] = dt.datetime.now().isoformat()
        handle.attrs["source_csv"] = str(csv_path)
        if nominal_dt > 0.0:
            handle.attrs["rate_hz"] = float(1.0 / nominal_dt)
        handle.attrs["action_dim"] = int(action_dim)
        env_meta = {
            "env_name": env_name,
            "type": 1,
            "env_kwargs": {"action_dim": action_dim, "obs_keys": list(obs_keys)},
        }
        handle.attrs["env_args"] = json.dumps(env_meta)
        data_group = handle.create_group("data")

        for idx, (obs_list, act_list, ts_list) in enumerate(episodes):
            group = data_group.create_group(f"demo_{idx}")
            obs_group = group.create_group("obs")
            keys = obs_list[0].keys()
            for key in keys:
                data = np.stack([step[key] for step in obs_list], axis=0)
                obs_group.create_dataset(key, data=data, compression="gzip")
            actions = np.stack(act_list, axis=0)
            group.create_dataset("actions", data=actions, compression="gzip")
            rewards = np.zeros(actions.shape[0], dtype=np.float32)
            group.create_dataset("rewards", data=rewards)
            dones = np.zeros(actions.shape[0], dtype=np.bool_)
            if dones.size:
                dones[-1] = True
            group.create_dataset("dones", data=dones)
            timestamps = np.asarray(ts_list, dtype=np.float64)
            group.create_dataset("timestamps", data=timestamps)
            duration = float(timestamps[-1] - timestamps[0]) if timestamps.size > 1 else 0.0
            group.attrs["num_samples"] = int(actions.shape[0])
            group.attrs["duration"] = duration


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    output_path = args.output.resolve() if args.output else csv_path.with_suffix(".hdf5")

    rows = list(iter_csv_rows(csv_path))
    if not rows:
        raise RuntimeError("CSV contains no data rows")

    header = set(rows[0].keys())

    if args.episode_column and args.episode_column not in header:
        raise ValueError(f"Episode column '{args.episode_column}' not found in CSV header")

    missing_columns: Dict[str, List[str]] = {}
    for obs_key in args.obs:
        required = OBS_REQUIRED_COLUMNS.get(obs_key, set())
        missing = sorted(col for col in required if col not in header)
        if missing:
            missing_columns[obs_key] = missing
    if missing_columns:
        problems = ", ".join(f"{key}: {cols}" for key, cols in missing_columns.items())
        raise ValueError(f"CSV missing required columns for observations -> {problems}")

    action_columns = args.action_columns
    if action_columns:
        missing_actions = [col for col in action_columns if col not in header]
        if missing_actions:
            raise ValueError(f"CSV missing requested action columns: {missing_actions}")
    action_dim = len(action_columns) if action_columns else args.action_dim

    episodes, nominal_dt = segment_episodes(
        rows=rows,
        obs_keys=args.obs,
        action_columns=action_columns,
        action_dim=action_dim,
        episode_column=args.episode_column,
        episode_gap_sec=args.episode_gap_sec,
        no_gap_split=args.no_gap_split,
    )

    if not episodes:
        raise RuntimeError("No episodes were assembled from the CSV contents")

    write_hdf5(
        output_path=output_path,
        csv_path=csv_path,
        episodes=episodes,
        obs_keys=args.obs,
        action_dim=action_dim,
        nominal_dt=nominal_dt,
        env_name=args.env_name,
    )

    total_steps = sum(len(episode[0]) for episode in episodes)
    print(
        f"Wrote {len(episodes)} episode(s) / {total_steps} samples to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
