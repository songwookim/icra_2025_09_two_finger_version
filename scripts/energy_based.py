#!/usr/bin/env python3
"""Energy-based analysis of a demonstration trial.

This script loads a synchronized demonstration CSV (default: trial_1.csv)
and computes basic energy metrics by integrating contact forces along the
end-effector displacement trajectory. The implementation mirrors the
paper's emphasis on interpreting puncture dynamics via force and motion
signals, providing a quick jump-off point for deeper energy-based models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


EPS = 1e-9


def fs_from_stamp(sec: pd.Series, nsec: pd.Series) -> float:
    """Infer sampling rate from ROS-style (sec, nsec) stamp columns."""
    if sec is None or nsec is None:
        return 0.0
    if len(sec) < 2 or len(nsec) < 2:
        return 0.0
    try:
        t = sec.to_numpy(dtype=float) + nsec.to_numpy(dtype=float) * 1e-9
    except Exception:  # pragma: no cover - robust fallback
        return 0.0

    diff = np.diff(t)
    diff = diff[np.isfinite(diff) & (diff > 0)]
    if diff.size == 0:
        return 0.0
    dt = np.median(diff)
    if dt <= 0:
        return 0.0
    return float(1.0 / dt)


def infer_rate_from_columns(df: pd.DataFrame, prefix: str) -> float:
    """Convenience wrapper that searches for <prefix>_stamp_(sec|nsec)."""
    sec_col = f"{prefix}_stamp_sec"
    nsec_col = f"{prefix}_stamp_nsec"
    if sec_col in df.columns and nsec_col in df.columns:
        return fs_from_stamp(df[sec_col], df[nsec_col])
    return 0.0


def load_trial(csv_path: Path) -> pd.DataFrame:
    """Load a demonstration CSV and validate key columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_force_cols = {
        "s2_fx",
        "s2_fy",
        "s2_fz",
        "s3_fx",
        "s3_fy",
        "s3_fz",
    }
    required_pose_cols = {"ee_px", "ee_py", "ee_pz"}

    missing = (required_force_cols | required_pose_cols) - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns in CSV: " + ", ".join(sorted(missing))
        )

    return df


def compute_contact_force(df: pd.DataFrame) -> np.ndarray:
    """Average the two force sensors and baseline-subtract using the first 10%."""
    f_s2 = df[["s2_fx", "s2_fy", "s2_fz"]].to_numpy(dtype=float)
    f_s3 = df[["s3_fx", "s3_fy", "s3_fz"]].to_numpy(dtype=float)

    forces = (f_s2 + f_s3) * 0.5
    rest_samples = max(1, int(round(0.1 * len(forces))))
    baseline = np.mean(forces[:rest_samples, :], axis=0, keepdims=True)
    return forces - baseline


def compute_displacement(df: pd.DataFrame) -> np.ndarray:
    """Compute EE displacement relative to the median of the first 10%."""
    positions = df[["ee_px", "ee_py", "ee_pz"]].to_numpy(dtype=float)
    rest_samples = max(1, int(round(0.1 * len(positions))))
    ref = np.median(positions[:rest_samples, :], axis=0, keepdims=True)
    return positions - ref


def integrate_work(
    force: np.ndarray,
    displacement: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate incremental mechanical work: W = ∫ F · dx.

    Returns
    -------
    cumulative_axis : np.ndarray
        Cumulative work per axis (shape: samples x 3).
    cumulative_total : np.ndarray
        Scalar cumulative work (samples,).
    """

    if force.shape != displacement.shape:
        raise ValueError("Force and displacement arrays must have matching shapes")

    if len(force) < 2:
        zero_axis = np.zeros_like(force)
        zero_total = np.zeros(force.shape[0], dtype=float)
        return zero_axis, zero_total

    delta_disp = np.diff(displacement, axis=0)
    mid_force = 0.5 * (force[1:, :] + force[:-1, :])

    incremental_axis = mid_force * delta_disp
    cumulative_axis = np.vstack(
        [np.zeros((1, force.shape[1])), np.cumsum(incremental_axis, axis=0)]
    )

    incremental_total = np.sum(incremental_axis, axis=1)
    cumulative_total = np.concatenate(
        [np.zeros(1, dtype=float), np.cumsum(incremental_total)]
    )

    return cumulative_axis, cumulative_total


def summarize_energy(cumulative_axis: np.ndarray, cumulative_total: np.ndarray) -> None:
    """Print summary statistics for cumulative work values."""
    labels = ["x", "y", "z"]
    for axis, label in enumerate(labels):
        axis_work = cumulative_axis[:, axis]
        print(
            f"Axis {label}: final work = {axis_work[-1]:.4f} N·m, "
            f"min = {np.min(axis_work):.4f}, max = {np.max(axis_work):.4f}"
        )

    print(
        f"Total work: final = {cumulative_total[-1]:.4f} N·m, "
        f"min = {np.min(cumulative_total):.4f}, max = {np.max(cumulative_total):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Energy-based inspection of a demonstration trial"
    )
    default_csv = Path("outputs") / "logs" / "20250929" / "trial_1.csv"
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=default_csv,
        help="Path to the synchronized demonstration CSV",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing energy summary statistics",
    )

    args = parser.parse_args()
    df = load_trial(args.csv_path)

    fs_force = infer_rate_from_columns(df, "s2")
    if fs_force <= 0:
        fs_force = infer_rate_from_columns(df, "s3")
    if fs_force <= 0:
        fs_force = infer_rate_from_columns(df, "ee")

    if fs_force > 0:
        print(f"Estimated sample rate: {fs_force:.3f} Hz")
    else:
        print("Sample rate could not be inferred from stamp columns.")

    forces = compute_contact_force(df)
    displacement = compute_displacement(df)

    cumulative_axis, cumulative_total = integrate_work(forces, displacement)

    if not args.no_summary:
        summarize_energy(cumulative_axis, cumulative_total)


if __name__ == "__main__":
    main()
