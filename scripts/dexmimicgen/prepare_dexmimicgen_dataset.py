#!/usr/bin/env python3
"""Convert midpoint-eccentricity CSV demonstrations into DexMimicGen-ready HDF5."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import h5py  # type: ignore
import numpy as np

DEFAULT_OBS_KEYS = ["ee_pos", "force", "deform", "emg"]
FORCE_COMPONENTS = ("fx", "fy", "fz", "tx", "ty", "tz")
EMG_COLUMNS = [f"emg_ch{i}" for i in range(1, 9)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare DexMimicGen demonstration dataset from CSV logs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="CSV files or directories that contain CSV demonstrations",
    )
    parser.add_argument("--output", required=True, type=Path, help="Destination HDF5 path")
    parser.add_argument("--material", required=True, help="Material label (e.g., hard, soft)")
    parser.add_argument("--target-ecc", type=float, default=0.5, help="Target eccentricity")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Acceptable absolute error around target eccentricity",
    )
    parser.add_argument(
        "--min-target-frames",
        type=int,
        default=10,
        help="Minimum number of samples that must satisfy the eccentricity tolerance",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Keep only the segment around the target eccentricity window",
    )
    parser.add_argument(
        "--pre-pad",
        type=int,
        default=20,
        help="Frames kept before the first in-tolerance index when cropping",
    )
    parser.add_argument(
        "--post-pad",
        type=int,
        default=20,
        help="Frames kept after the last in-tolerance index when cropping",
    )
    parser.add_argument(
        "--obs",
        nargs="+",
        default=DEFAULT_OBS_KEYS,
        help="Observation keys to export",
    )
    parser.add_argument(
        "--action-columns",
        nargs="+",
        default=None,
        help="CSV column names to treat as actions",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=3,
        help="Action dimensionality when explicit columns are not provided",
    )
    parser.add_argument(
        "--allow-missing-target",
        action="store_true",
        help="Include demos even if they never reach the eccentricity tolerance",
    )
    parser.add_argument(
        "--metadata",
        nargs="*",
        default=None,
        help="Additional key=value metadata pairs to store at the file level",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when breaking ties during cropping",
    )
    return parser.parse_args()


def collect_csv_files(paths: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.csv") if p.is_file()))
        elif path.is_file() and path.suffix.lower() == ".csv":
            files.append(path)
    unique_files = []
    seen = set()
    for f in files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    return unique_files


def iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        for row in reader:
            yield row


def safe_float(value: Optional[str], default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_timestamp(row: Dict[str, str]) -> float:
    sec = safe_float(row.get("t_sec"), 0.0)
    nsec = safe_float(row.get("t_nanosec"), 0.0)
    return sec + (nsec * 1e-9)


def build_observation(row: Dict[str, str], obs_keys: Sequence[str]) -> Dict[str, np.ndarray]:
    obs: Dict[str, np.ndarray] = {}
    if "ee_pos" in obs_keys:
        obs["ee_pos"] = np.asarray(
            [safe_float(row.get("ee_px")), safe_float(row.get("ee_py")), safe_float(row.get("ee_pz"))],
            dtype=np.float32,
        )
    if "ee_quat" in obs_keys:
        obs["ee_quat"] = np.asarray(
            [
                safe_float(row.get("ee_qx"), default=0.0),
                safe_float(row.get("ee_qy"), default=0.0),
                safe_float(row.get("ee_qz"), default=0.0),
                safe_float(row.get("ee_qw"), default=1.0),
            ],
            dtype=np.float32,
        )
    if "force" in obs_keys or "force_s2" in obs_keys or "force_s3" in obs_keys:
        s2 = np.asarray([safe_float(row.get(f"s2_{c}")) for c in FORCE_COMPONENTS], dtype=np.float32)
        s3 = np.asarray([safe_float(row.get(f"s3_{c}")) for c in FORCE_COMPONENTS], dtype=np.float32)
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


def build_action(row: Dict[str, str], action_columns: Optional[Sequence[str]], action_dim: int) -> np.ndarray:
    if action_columns:
        return np.asarray([safe_float(row.get(col)) for col in action_columns], dtype=np.float32)
    return np.zeros(action_dim, dtype=np.float32)


def enforce_monotonic(timestamps: np.ndarray) -> np.ndarray:
    adjusted = timestamps.copy()
    for i in range(1, adjusted.shape[0]):
        if adjusted[i] <= adjusted[i - 1]:
            adjusted[i] = math.nextafter(adjusted[i - 1], float("inf"))
    return adjusted


def crop_indices(
    indices: np.ndarray,
    length: int,
    pre: int,
    post: int,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    base_start = max(int(indices[0]) - pre, 0)
    base_end = min(int(indices[-1]) + post + 1, length)
    jittered_start = base_start
    jittered_end = base_end
    if pre > 0:
        head_room = min(int(indices[0]), pre)
        if head_room > 0:
            jitter = int(rng.integers(0, head_room + 1))
            jittered_start = max(int(indices[0]) - jitter, 0)
    if post > 0:
        tail_room = min(length - int(indices[-1]) - 1, post)
        if tail_room > 0:
            jitter = int(rng.integers(0, tail_room + 1))
            jittered_end = min(int(indices[-1]) + jitter + 1, length)
    return jittered_start, jittered_end


def parse_metadata_args(pairs: Optional[Sequence[str]]) -> Dict[str, str]:
    if not pairs:
        return {}
    meta: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Metadata entry must be key=value, got '{item}'")
        key, value = item.split("=", 1)
        meta[key.strip()] = value.strip()
    return meta


def assemble_demo(
    path: Path,
    obs_keys: Sequence[str],
    action_columns: Optional[Sequence[str]],
    action_dim: int,
    target_ecc: float,
    tolerance: float,
    crop: bool,
    pre_pad: int,
    post_pad: int,
    min_target: int,
    allow_missing: bool,
    rng: np.random.Generator,
) -> Optional[Dict[str, object]]:
    rows = list(iter_csv_rows(path))
    if not rows:
        return None

    timestamps = np.asarray([compute_timestamp(row) for row in rows], dtype=np.float64)
    timestamps = enforce_monotonic(timestamps)
    obs_buffers: Dict[str, List[np.ndarray]] = {key: [] for key in obs_keys}
    actions: List[np.ndarray] = []
    ecc_series = np.asarray([safe_float(row.get("deform_ecc")) for row in rows], dtype=np.float32)

    for row in rows:
        obs = build_observation(row, obs_keys)
        for key in obs_keys:
            if key not in obs:
                raise ValueError(f"Observation key '{key}' missing in {path.name}")
            obs_buffers[key].append(obs[key])
        actions.append(build_action(row, action_columns, action_dim))

    obs_arrays = {key: np.stack(values, axis=0) for key, values in obs_buffers.items()}
    actions_array = np.stack(actions, axis=0)

    target_mask = np.where(np.abs(ecc_series - target_ecc) <= tolerance)[0]
    if target_mask.size < min_target:
        if not allow_missing:
            return None
    crop_slice = slice(0, len(rows))
    if crop and target_mask.size > 0:
        start, end = crop_indices(target_mask, len(rows), pre_pad, post_pad, rng)
        crop_slice = slice(start, end)

    for key in obs_arrays:
        obs_arrays[key] = obs_arrays[key][crop_slice]
    actions_array = actions_array[crop_slice]
    timestamps = timestamps[crop_slice]
    ecc_series = ecc_series[crop_slice]

    duration = float(timestamps[-1] - timestamps[0]) if timestamps.size > 1 else 0.0
    median_dt = float(np.median(np.diff(timestamps))) if timestamps.size > 1 else 0.0

    demo_meta = {
        "source_csv": str(path),
        "num_samples": int(timestamps.shape[0]),
        "duration": duration,
        "median_dt": median_dt,
        "target_ecc": target_ecc,
        "tolerance": tolerance,
        "ecc_min": float(ecc_series.min(initial=0.0)),
        "ecc_max": float(ecc_series.max(initial=0.0)),
        "ecc_mean": float(ecc_series.mean()) if ecc_series.size else 0.0,
        "hit_frames": int(target_mask.size),
        "cropped": bool(crop and target_mask.size > 0),
    }
    return {
        "obs": obs_arrays,
        "actions": actions_array.astype(np.float32, copy=False),
        "timestamps": timestamps,
        "metadata": demo_meta,
    }


def write_hdf5(
    output: Path,
    demos: List[Dict[str, object]],
    obs_keys: Sequence[str],
    action_dim: int,
    material: str,
    target_ecc: float,
    tolerance: float,
    file_metadata: Dict[str, str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.attrs["material"] = material
        handle.attrs["target_ecc"] = target_ecc
        handle.attrs["tolerance"] = tolerance
        handle.attrs["num_demos"] = len(demos)
        handle.attrs["action_dim"] = action_dim
        handle.attrs["obs_keys"] = json.dumps(list(obs_keys))
        if file_metadata:
            handle.attrs["user_metadata"] = json.dumps(file_metadata)
        source_list = [cast(Dict[str, Any], demo["metadata"])["source_csv"] for demo in demos]
        handle.attrs["source_csvs"] = json.dumps(source_list)
        env_meta = {
            "env_name": "TwoFingerPuncture",
            "type": 1,
            "env_kwargs": {"material": material, "target_ecc": target_ecc, "obs_keys": list(obs_keys)},
        }
        handle.attrs["env_args"] = json.dumps(env_meta)

        data_group = handle.create_group("data")
        for idx, demo in enumerate(demos):
            group = data_group.create_group(f"demo_{idx}")
            obs_group = group.create_group("obs")
            obs_dict = cast(Dict[str, np.ndarray], demo["obs"])
            for key, array in obs_dict.items():
                obs_group.create_dataset(key, data=array, compression="gzip")
            actions = cast(np.ndarray, demo["actions"])
            group.create_dataset("actions", data=actions, compression="gzip")
            timestamps = cast(np.ndarray, demo["timestamps"])
            group.create_dataset("timestamps", data=timestamps)
            dones = np.zeros(actions.shape[0], dtype=np.bool_)
            if dones.size:
                dones[-1] = True
            group.create_dataset("dones", data=dones)
            rewards = np.zeros(actions.shape[0], dtype=np.float32)
            group.create_dataset("rewards", data=rewards)
            group.attrs["metadata"] = json.dumps(cast(Dict[str, Any], demo["metadata"]))


def main() -> None:
    args = parse_args()
    csv_files = collect_csv_files(args.inputs)
    if not csv_files:
        raise SystemExit("No CSV files found in the provided paths")

    file_metadata = parse_metadata_args(args.metadata)

    rng = np.random.default_rng(args.seed)

    demos: List[Dict[str, object]] = []
    skipped: List[str] = []
    for csv_path in csv_files:
        demo = assemble_demo(
            path=csv_path,
            obs_keys=args.obs,
            action_columns=args.action_columns,
            action_dim=args.action_dim,
            target_ecc=args.target_ecc,
            tolerance=args.tolerance,
            crop=args.crop,
            pre_pad=args.pre_pad,
            post_pad=args.post_pad,
            min_target=args.min_target_frames,
            allow_missing=args.allow_missing_target,
            rng=rng,
        )
        if demo is None:
            skipped.append(str(csv_path))
            continue
        demos.append(demo)

    if not demos:
        raise SystemExit("All CSV inputs were skipped. Check tolerance or provide --allow-missing-target.")

    write_hdf5(
        output=args.output,
        demos=demos,
        obs_keys=args.obs,
        action_dim=args.action_dim,
        material=args.material,
        target_ecc=args.target_ecc,
        tolerance=args.tolerance,
        file_metadata=file_metadata,
    )

    print(f"Wrote {len(demos)} demos to {args.output}")
    if skipped:
        print("Skipped demos (missing target hits):")
        for item in skipped:
            print(f" - {item}")


if __name__ == "__main__":
    main()
