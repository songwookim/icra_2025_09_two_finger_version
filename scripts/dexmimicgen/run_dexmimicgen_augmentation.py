#!/usr/bin/env python3
"""Augment DexMimicGen datasets using the official or a local fallback backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import h5py  # type: ignore
import numpy as np

try:  # pragma: no cover - optional dependency
    from dexmimicgen import DexAugmentor  # type: ignore

    HAS_DEXMIMICGEN = True
except ImportError:  # pragma: no cover - optional dependency
    DexAugmentor = None  # type: ignore
    HAS_DEXMIMICGEN = False


Demo = Dict[str, Any]
ObservationDict = Dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic demonstrations compatible with DexMimicGen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="Input HDF5 produced by prepare_dexmimicgen_dataset.py")
    parser.add_argument("--output", required=True, type=Path, help="Destination HDF5 for augmented demos")
    parser.add_argument(
        "--backend",
        choices=["auto", "dexmimicgen", "local"],
        default="auto",
        help="Augmentation backend to use",
    )
    parser.add_argument("--per-demo", type=int, default=50, help="Synthetic demos to generate per input demo")
    parser.add_argument("--include-original", action="store_true", help="Copy original demos into the output file as well")
    parser.add_argument("--pos-noise-std", type=float, default=0.002, help="Gaussian noise (m) added to ee_pos")
    parser.add_argument("--force-scale-min", type=float, default=0.9, help="Minimum multiplicative scale for force components")
    parser.add_argument("--force-scale-max", type=float, default=1.1, help="Maximum multiplicative scale for force components")
    parser.add_argument("--ecc-noise-std", type=float, default=0.02, help="Noise applied to eccentricity channel")
    parser.add_argument("--emg-noise-std", type=float, default=3.0, help="Gaussian noise for EMG channels")
    parser.add_argument("--action-noise-std", type=float, default=0.05, help="Action noise standard deviation")
    parser.add_argument("--temporal-scale-min", type=float, default=0.85, help="Lower bound for temporal stretching")
    parser.add_argument("--temporal-scale-max", type=float, default=1.15, help="Upper bound for temporal stretching")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for local backend")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON or YAML file with augmentation parameters that override CLI defaults",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do everything except writing the output file")
    return parser.parse_args()


def enforce_monotonic(timestamps: np.ndarray) -> np.ndarray:
    adjusted = timestamps.copy()
    for i in range(1, adjusted.shape[0]):
        if adjusted[i] <= adjusted[i - 1]:
            adjusted[i] = np.nextafter(adjusted[i - 1], np.inf)
    return adjusted


def load_dataset(path: Path) -> Tuple[List[Demo], Dict[str, Any]]:
    demos: List[Demo] = []
    root_attrs: Dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key, value in handle.attrs.items():
            if isinstance(value, bytes):
                root_attrs[key] = value.decode("utf-8")
            else:
                root_attrs[key] = value
        root_attrs.setdefault("source_dataset", str(path))
        data_group = handle.get("data")
        if data_group is None:
            raise RuntimeError(f"Input file {path} has no 'data' group")
        for name in sorted(data_group.keys()):
            group = data_group[name]
            obs_group = group["obs"]
            obs: ObservationDict = {key: obs_group[key][:].astype(np.float32) for key in obs_group.keys()}
            actions = group["actions"][:].astype(np.float32)
            timestamps = group["timestamps"][:].astype(np.float64)
            timestamps = enforce_monotonic(timestamps)
            meta_raw = group.attrs.get("metadata")
            if isinstance(meta_raw, bytes):
                meta_raw = meta_raw.decode("utf-8")
            metadata: Dict[str, Any]
            if isinstance(meta_raw, str):
                try:
                    metadata = json.loads(meta_raw)
                except json.JSONDecodeError:
                    metadata = {"raw_metadata": meta_raw}
            else:
                metadata = {}
            metadata.setdefault("source_demo", name)
            demos.append(
                {
                    "name": name,
                    "obs": obs,
                    "actions": actions,
                    "timestamps": timestamps,
                    "metadata": metadata,
                }
            )
    return demos, root_attrs


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:  # pragma: no cover - optional dependency
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit("PyYAML is required to parse YAML config files") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Config file must define a mapping")
    return data


class LocalDexAugmentor:
    """Lightweight augmentation backend inspired by DexMimicGen."""

    def __init__(
        self,
        obs_keys: Sequence[str],
        pos_noise_std: float,
        force_scale: Tuple[float, float],
        ecc_noise_std: float,
        emg_noise_std: float,
        action_noise_std: float,
        temporal_scale: Tuple[float, float],
        seed: int,
    ) -> None:
        self.obs_keys = list(obs_keys)
        self.pos_noise_std = pos_noise_std
        self.force_scale = force_scale
        self.ecc_noise_std = ecc_noise_std
        self.emg_noise_std = emg_noise_std
        self.action_noise_std = action_noise_std
        self.temporal_scale = temporal_scale
        self.rng = np.random.default_rng(seed)

    def augment(self, demo: Demo, count: int) -> List[Demo]:
        outputs: List[Demo] = []
        base_obs = cast(ObservationDict, demo["obs"])
        base_actions = cast(np.ndarray, demo["actions"])
        base_ts = cast(np.ndarray, demo["timestamps"])
        base_meta = cast(Dict[str, Any], demo["metadata"]).copy()

        for _ in range(count):
            temporal_scale = float(self.rng.uniform(self.temporal_scale[0], self.temporal_scale[1]))
            new_length = max(8, int(round(base_ts.shape[0] * temporal_scale)))
            new_ts = np.linspace(base_ts[0], base_ts[-1], new_length, dtype=np.float64)
            new_ts = enforce_monotonic(new_ts) - new_ts[0]

            warped_obs: ObservationDict = {}
            for key, array in base_obs.items():
                warped_obs[key] = self._resample(array, base_ts, new_ts)

            actions_aug = self._resample(base_actions, base_ts, new_ts)

            if "ee_pos" in warped_obs and self.pos_noise_std > 0.0:
                warped_obs["ee_pos"] += self.rng.normal(
                    0.0, self.pos_noise_std, size=warped_obs["ee_pos"].shape
                ).astype(np.float32)

            force_scale = float(self.rng.uniform(self.force_scale[0], self.force_scale[1]))
            for key in warped_obs:
                if key.startswith("force"):
                    warped_obs[key][:, :3] *= force_scale

            if "deform" in warped_obs and self.ecc_noise_std > 0.0:
                ecc_noise = self.rng.normal(0.0, self.ecc_noise_std, size=warped_obs["deform"].shape[0]).astype(
                    np.float32
                )
                warped_obs["deform"][:, 1] = np.clip(warped_obs["deform"][:, 1] + ecc_noise, 0.0, 1.0)

            if "emg" in warped_obs and self.emg_noise_std > 0.0:
                warped_obs["emg"] += self.rng.normal(
                    0.0, self.emg_noise_std, size=warped_obs["emg"].shape
                ).astype(np.float32)

            if self.action_noise_std > 0.0:
                actions_aug += self.rng.normal(0.0, self.action_noise_std, size=actions_aug.shape).astype(np.float32)
            actions_aug = actions_aug.astype(np.float32, copy=False)

            metadata = base_meta.copy()
            metadata.update(
                {
                    "augmented": True,
                    "source_demo": demo.get("name", metadata.get("source_demo", "unknown")),
                    "temporal_scale": temporal_scale,
                    "force_scale": force_scale,
                    "pos_noise_std": self.pos_noise_std,
                    "ecc_noise_std": self.ecc_noise_std,
                    "emg_noise_std": self.emg_noise_std,
                    "action_noise_std": self.action_noise_std,
                    "num_samples": new_length,
                    "duration": float(new_ts[-1]) if new_ts.size > 0 else 0.0,
                    "median_dt": float(np.median(np.diff(new_ts))) if new_ts.size > 1 else 0.0,
                    "rng_snapshot": int(self.rng.integers(0, 2**31 - 1)),
                }
            )

            outputs.append(
                {
                    "obs": warped_obs,
                    "actions": actions_aug,
                    "timestamps": new_ts,
                    "metadata": metadata,
                }
            )
        return outputs

    def _resample(self, data: np.ndarray, old_ts: np.ndarray, new_ts: np.ndarray) -> np.ndarray:
        old_ts = enforce_monotonic(old_ts)
        if data.ndim == 1:
            res = np.interp(new_ts, old_ts, data).astype(np.float32)
            return res
        channels = [np.interp(new_ts, old_ts, data[:, idx]) for idx in range(data.shape[1])]
        return np.stack(channels, axis=1).astype(np.float32)


def maybe_merge_config(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    for key, value in config.items():
        cli_key = key.replace("-", "_")
        if hasattr(args, cli_key):
            setattr(args, cli_key, value)
    return args


def choose_backend(args: argparse.Namespace) -> str:
    if args.backend == "auto":
        return "dexmimicgen" if HAS_DEXMIMICGEN else "local"
    if args.backend == "dexmimicgen" and not HAS_DEXMIMICGEN:
        raise SystemExit("dexmimicgen package is not installed; choose --backend local or install DexMimicGen")
    return args.backend


def run_official_backend(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    demos: List[Demo],
) -> None:
    if DexAugmentor is None:  # pragma: no cover
        raise SystemExit("dexmimicgen package missing; cannot use official backend")
    config = {
        "per_demo": args.per_demo,
        "seed": args.seed,
        "include_original": args.include_original,
    }
    augmentor: Any
    try:  # pragma: no cover - depends on upstream API
        augmentor = DexAugmentor(config=config)
    except TypeError:
        try:
            augmentor = DexAugmentor(**config)
        except Exception as exc:
            raise SystemExit(f"Unable to initialise DexAugmentor automatically: {exc}") from exc

    demo_paths = [demo["metadata"].get("source_csv", demo["name"]) for demo in demos]
    candidate_methods = [
        "generate_dataset",
        "augment_dataset",
        "generate",
        "augment",
    ]
    method = None
    for name in candidate_methods:
        if hasattr(augmentor, name):
            method = getattr(augmentor, name)
            break
    if method is None:
        raise SystemExit(
            "DexAugmentor API is unknown. Please update run_dexmimicgen_augmentation.py with the correct call."
        )

    try:  # pragma: no cover - upstream dependent
        method(
            input_dataset=str(input_path),
            output_dataset=str(output_path),
            per_demo=args.per_demo,
            demo_paths=demo_paths,
        )
    except TypeError:
        method(str(input_path), str(output_path))
    print(f"[dexmimicgen] Delegated augmentation to DexAugmentor -> {output_path}")


def write_dataset(
    output: Path,
    base_attrs: Dict[str, Any],
    obs_keys: Sequence[str],
    action_dim: int,
    backend: str,
    augmented: List[Demo],
    originals: Optional[List[Demo]],
    per_demo: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        for key, value in base_attrs.items():
            if key in {"num_demos"}:
                continue
            handle.attrs[key] = value
        handle.attrs["source_dataset"] = base_attrs.get("source_dataset", "")
        handle.attrs["augmentation_backend"] = backend
        handle.attrs["num_original"] = len(originals) if originals else 0
        handle.attrs["num_augmented"] = len(augmented)
        handle.attrs["num_demos"] = (len(originals) if originals else 0) + len(augmented)
        handle.attrs["obs_keys"] = json.dumps(list(obs_keys))
        handle.attrs["action_dim"] = action_dim
        augment_cfg = {"backend": backend, "per_demo": per_demo}
        handle.attrs["augmentation_config"] = json.dumps(augment_cfg)

        env_args_raw = base_attrs.get("env_args")
        if isinstance(env_args_raw, (str, bytes)):
            try:
                env_args = json.loads(env_args_raw.decode("utf-8") if isinstance(env_args_raw, bytes) else env_args_raw)
            except json.JSONDecodeError:
                env_args = {"raw": env_args_raw}
        else:
            env_args = None
        if isinstance(env_args, dict):
            env_args_dict = cast(Dict[str, Any], env_args)
            env_args_dict["augmentation_backend"] = backend
            env_args_dict["num_augmented"] = len(augmented)
            handle.attrs["env_args"] = json.dumps(env_args_dict)

        data_group = handle.create_group("data")
        index = 0

        def _write_demo(demo: Demo, idx: int) -> None:
            group = data_group.create_group(f"demo_{idx:05d}")
            obs_group = group.create_group("obs")
            obs = cast(ObservationDict, demo["obs"])
            for key, array in obs.items():
                obs_group.create_dataset(key, data=array.astype(np.float32), compression="gzip")
            actions = cast(np.ndarray, demo["actions"])
            group.create_dataset("actions", data=actions.astype(np.float32), compression="gzip")
            timestamps = cast(np.ndarray, demo["timestamps"])
            group.create_dataset("timestamps", data=timestamps.astype(np.float64))
            dones = np.zeros(actions.shape[0], dtype=np.bool_)
            if dones.size:
                dones[-1] = True
            group.create_dataset("dones", data=dones)
            rewards = np.zeros(actions.shape[0], dtype=np.float32)
            group.create_dataset("rewards", data=rewards)
            metadata = dict(cast(Dict[str, Any], demo.get("metadata", {})))
            metadata.setdefault("source_demo", demo.get("name", metadata.get("source_demo", "unknown")))
            metadata.setdefault("augmented", False)
            group.attrs["metadata"] = json.dumps(metadata)

        if originals:
            for demo in originals:
                _write_demo(demo, index)
                index += 1
        for demo in augmented:
            _write_demo(demo, index)
            index += 1

    print(f"Wrote augmented dataset -> {output} (demos={index})")


def run_local_backend(
    args: argparse.Namespace,
    demos: List[Demo],
    root_attrs: Dict[str, Any],
    output_path: Path,
) -> None:
    raw_obs_keys = root_attrs.get("obs_keys")
    if isinstance(raw_obs_keys, str):
        obs_keys = json.loads(raw_obs_keys)
    elif isinstance(raw_obs_keys, (list, tuple)):
        obs_keys = list(raw_obs_keys)
    else:
        obs_keys = ["ee_pos", "force", "deform", "emg"]

    raw_action_dim = root_attrs.get("action_dim", 3)
    action_dim = int(raw_action_dim)

    augmentor = LocalDexAugmentor(
        obs_keys=obs_keys,
        pos_noise_std=args.pos_noise_std,
        force_scale=(args.force_scale_min, args.force_scale_max),
        ecc_noise_std=args.ecc_noise_std,
        emg_noise_std=args.emg_noise_std,
        action_noise_std=args.action_noise_std,
        temporal_scale=(args.temporal_scale_min, args.temporal_scale_max),
        seed=args.seed,
    )

    augmented: List[Demo] = []
    for demo in demos:
        augmented.extend(augmentor.augment(demo, args.per_demo))

    if args.dry_run:
        print(f"[dry-run] Prepared {len(augmented)} synthetic demos")
        return

    originals = demos if args.include_original else None
    write_dataset(
        output=output_path,
        base_attrs=root_attrs,
        obs_keys=obs_keys,
        action_dim=action_dim,
        backend="local",
        augmented=augmented,
        originals=originals,
        per_demo=args.per_demo,
    )


def main() -> None:
    args = parse_args()
    config_overrides = load_config(args.config)
    args = maybe_merge_config(args, config_overrides)

    backend = choose_backend(args)
    demos, root_attrs = load_dataset(args.input)

    if backend == "dexmimicgen":
        if args.dry_run:
            print("[dry-run] Would invoke DexMimicGen official backend")
            return
        run_official_backend(
            input_path=args.input,
            output_path=args.output,
            args=args,
            demos=demos,
        )
        return

    run_local_backend(
        args=args,
        demos=demos,
        root_attrs=root_attrs,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
