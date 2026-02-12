#!/usr/bin/env python3
"""Compute T_F, projection matrices, and stiffness mapping from a single trial."""

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

import matplotlib

matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


EPS = 1e-8


def median_center(signal: np.ndarray, portion: float = 0.1) -> np.ndarray:
    """Remove the median baseline computed over the initial portion of samples."""
    if signal.ndim != 2 or signal.size == 0:
        return signal
    span = max(1, int(round(portion * signal.shape[0])))
    baseline = np.median(signal[:span, :], axis=0, keepdims=True)
    return signal - baseline


def fs_from_stamp(sec: pd.Series, nsec: pd.Series) -> float:
    """Infer sampling rate from synchronized stamp columns."""
    if sec is None or nsec is None:
        return 0.0
    if len(sec) < 2 or len(nsec) < 2:
        return 0.0
    try:
        t = sec.to_numpy(dtype=float) + nsec.to_numpy(dtype=float) * 1e-9
    except Exception:
        return 0.0
    diff = np.diff(t)
    diff = diff[np.isfinite(diff) & (diff > 0)]
    if diff.size == 0:
        return 0.0
    dt = np.median(diff)
    if dt <= 0:
        return 0.0
    return float(1.0 / dt)


def apply_notch_filters(emg: np.ndarray, fs: float, line_freq: float = 60.0, harmonics: Tuple[int, ...] = (1, 2, 3), Q: float = 30.0) -> np.ndarray:
    """Apply sequential notch filters to suppress line noise and harmonics."""
    if fs <= 0 or line_freq <= 0:
        return emg
    filtered = emg
    nyquist = 0.5 * fs
    for h in harmonics:
        freq = h * line_freq
        if freq >= nyquist:
            break
        w0 = freq / nyquist
        try:
            b, a = iirnotch(w0, Q)
            filtered = filtfilt(b, a, filtered, axis=0)
        except ValueError:
            # iirnotch can fail if w0 is too close to 0 or 1
            continue
    return filtered


def design_lowpass(fs: float, cutoff: Optional[float], order: int = 5) -> Optional[np.ndarray]:
    if fs <= 0 or cutoff is None or cutoff <= 0 or order <= 0:
        return None
    max_cutoff = 0.45 * fs
    cut = min(cutoff, max_cutoff)
    if cut <= 0:
        return None
    sos = butter(order, cut, btype='low', fs=fs, output='sos')
    return np.asarray(sos, dtype=float)


def infer_rate_from_columns(df: pd.DataFrame, prefix: str) -> float:
    sec_col = f'{prefix}_stamp_sec'
    nsec_col = f'{prefix}_stamp_nsec'
    if sec_col in df.columns and nsec_col in df.columns:
        return fs_from_stamp(df[sec_col], df[nsec_col])
    return 0.0

def load_trial(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV not found: {csv_path}')
    df = pd.read_csv(csv_path)
    required_cols = {
        's2_fx', 's2_fy', 's2_fz',
        's3_fx', 's3_fy', 's3_fz',
        'ee_px', 'ee_py', 'ee_pz',
        'deform_ecc',
    }
    emg_cols = {f'emg_ch{i}' for i in range(1, 9)}
    missing = (required_cols | emg_cols) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    return df


def compute_emg_variants(
    emg_raw: np.ndarray,
    fs: float,
    line_freq: Optional[float] = 60.0,
    envelope_window_sec: float = 0.15,
    lowpass_cutoff: Optional[float] = None,
    lowpass_orders: Optional[List[int]] = None,
    fs_overrides: Optional[List[float]] = None,
    bandpass_range: Tuple[float, float] = (90.0, 450.0),
    bandpass_order: int = 4,
) -> OrderedDict[str, np.ndarray]:
    """Return multiple EMG representations for comparison."""

    if emg_raw.ndim != 2:
        raise ValueError('EMG must be 2-D (samples x channels)')

    samples = emg_raw.shape[0]
    rest_samples = max(1, int(0.1 * samples))
    baseline = np.median(emg_raw[:rest_samples, :], axis=0, keepdims=True)
    emg_centered = emg_raw - baseline

    if line_freq:
        emg_denoised = apply_notch_filters(emg_centered, fs, line_freq=line_freq)
    else:
        emg_denoised = emg_centered

    variants: OrderedDict[str, np.ndarray] = OrderedDict()

    variants['raw'] = emg_raw.copy()
    variants['raw_centered'] = emg_centered.copy()

    band_low, band_high = bandpass_range
    bandpassed = emg_denoised.copy()
    if fs > 0 and band_high > band_low and bandpass_order > 0:
        nyquist = 0.5 * fs
        low = max(band_low, 1e-3)
        high = min(band_high, nyquist - 1e-3)
        if high > low:
            sos_band = butter(
                bandpass_order,
                [low, high],
                btype='bandpass',
                fs=fs,
                output='sos',
            )
            bandpassed = sosfiltfilt(sos_band, emg_denoised, axis=0)
    variants['bandpass'] = bandpassed.copy()

    # Reference pipeline: invert → band-pass (20–500 Hz) → 100 ms moving-average
    pipeline_band = np.abs(-emg_denoised)
    if fs > 0:
        ref_low, ref_high = 20.0, min(500.0, 0.49 * fs)
        if ref_high > ref_low:
            sos_ref = butter(
                max(bandpass_order, 1),
                [ref_low, ref_high],
                btype='bandpass',
                fs=fs,
                output='sos',
            )
            filtered = sosfiltfilt(sos_ref, -emg_denoised, axis=0)
            pipeline_band = np.abs(filtered)
        window_samples = max(1, int(round(0.1 * fs)))
        if window_samples > 1:
            pipeline_band = uniform_filter1d(pipeline_band, size=window_samples, axis=0, mode='nearest')
    variants['rectified_bandpass_ma'] = pipeline_band

    orders = sorted({order for order in (lowpass_orders or [5]) if order and order > 0})
    if not orders:
        orders = [5]

    fs_candidates: List[float] = []
    if fs > 0:
        fs_candidates.append(fs)
    for fs_override in (fs_overrides or []):
        if fs_override and fs_override > 0:
            if not any(np.isclose(fs_override, existing) for existing in fs_candidates):
                fs_candidates.append(fs_override)
    if not fs_candidates:
        fs_candidates.append(fs if fs > 0 else 1000.0)

    base_order = orders[0]
    base_fs = fs_candidates[0]

    for fs_eff in fs_candidates:
        for order in orders:
            sos_low = design_lowpass(fs_eff, lowpass_cutoff, order=order)
            if sos_low is not None:
                filtered = sosfiltfilt(sos_low, emg_denoised, axis=0)
            else:
                filtered = emg_denoised.copy()

            smooth_window_sec = max(envelope_window_sec * 2.0, 0.25)
            smooth_window = max(1, int(round(smooth_window_sec * max(fs_eff, 1.0))))
            if smooth_window > 1:
                smoothed = uniform_filter1d(filtered, size=smooth_window, axis=0, mode='nearest')
            else:
                smoothed = filtered.copy()

            if np.isclose(fs_eff, base_fs) and order == base_order:
                variants['lowpass'] = filtered
                variants['lowpass_smooth'] = smoothed
            else:
                suffix = f"_o{order}_fs{fs_eff:g}"
                variants[f'lowpass{suffix}'] = filtered
                variants[f'lowpass_smooth{suffix}'] = smoothed

    return variants


def compute_force_variants(df: pd.DataFrame, fs_s2: float, fs_s3: float) -> Tuple[OrderedDict[str, np.ndarray], float]:
    """Return multiple force representations from each sensor."""

    forces_s2 = df[['s2_fx', 's2_fy', 's2_fz']].to_numpy(dtype=float)
    forces_s3 = df[['s3_fx', 's3_fy', 's3_fz']].to_numpy(dtype=float)

    s2_centered = median_center(forces_s2)
    s3_centered = median_center(forces_s3)

    variants: OrderedDict[str, np.ndarray] = OrderedDict()
    variants['s2_signed'] = s2_centered
    variants['s2_abs'] = np.abs(s2_centered)
    variants['s3_signed'] = s3_centered
    variants['s3_abs'] = np.abs(s3_centered)

    valid_rates = [rate for rate in (fs_s2, fs_s3) if rate > 0]
    fs_combined = float(np.median(valid_rates)) if valid_rates else 0.0
    return variants, fs_combined


def compute_tf(P: np.ndarray, F: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Solve T_F from F = T_F · P using ridge-regularized least squares."""
    PPt = P @ P.T
    reg_eye = reg * np.eye(PPt.shape[0])
    T_F = F @ P.T @ np.linalg.inv(PPt + reg_eye)
    return T_F


def compute_projection_from_tf(T_F: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Compute H_F = T_F^T (T_F T_F^T)^{-1} T_F."""
    TF_TF_T = T_F @ T_F.T
    reg_eye = reg * np.eye(TF_TF_T.shape[0])
    inv_term = np.linalg.inv(TF_TF_T + reg_eye)
    H_F = T_F.T @ inv_term @ T_F
    return H_F


def compute_tk_from_projection(H_F: np.ndarray, target_rank: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Recover T_K rows spanning the complementary subspace of H_F via eigen-decomposition."""
    dim = H_F.shape[0]
    H_K = np.eye(dim) - H_F

    eigvals, eigvecs = np.linalg.eigh(H_K)
    idx = np.argsort(eigvals)[::-1]
    significant = [i for i in idx if eigvals[i] > 1e-6]
    if len(significant) < target_rank:
        raise ValueError('Complementary subspace has insufficient rank for T_K')

    basis_idx = significant[:target_rank]
    basis = eigvecs[:, basis_idx].T
    T_K = basis

    return T_K, H_K


def smooth_stiffness(K: np.ndarray, fs: float, cutoff: float = 2.0, window_sec: float = 0.5) -> np.ndarray:
    """Apply low-pass filtering and moving-average smoothing to stiffness trajectories."""

    if K.ndim != 2:
        return K
    smoothed = K.copy()
    if fs > 0 and cutoff > 0:
        cutoff_adj = min(cutoff, 0.45 * fs - 1e-3)
        if cutoff_adj > 0.2:
            sos = butter(2, cutoff_adj, btype='low', fs=fs, output='sos')
            smoothed = sosfiltfilt(sos, smoothed, axis=1)

    window_samples = max(1, int(round(window_sec * max(fs, 1.0))))
    if window_samples > 1:
        smoothed = uniform_filter1d(smoothed, size=window_samples, axis=1, mode='nearest')
    return smoothed


def find_latest_csv(search_root: Path) -> Optional[Path]:
    if not search_root.exists():
        return None
    candidates = list(search_root.rglob('*.csv'))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description='Compute T_F, H_F, T_K, and K from a trial log')
    parser.add_argument('csv_path', nargs='?', help='Path to trial CSV; omit to auto-select latest under outputs/logs')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting (useful for headless runs)', default=False)
    parser.add_argument('--line-freq', type=float, default=60.0, help='Power-line frequency to notch from EMG (set 0 to disable)')
    parser.add_argument('--emg-envelope-window', type=float, default=0.15, help='EMG envelope smoothing window (seconds)')
    parser.add_argument('--emg-lowpass', type=float, default=10.0, help='Optional low-pass cutoff (Hz) applied to notch-filtered EMG (set 0 to disable)')
    parser.add_argument('--emg-lowpass-multi', type=str, default='', help='Comma-separated additional low-pass cutoffs (Hz) for comparison')
    parser.add_argument('--emg-lowpass-orders', type=str, default='5', help='Comma-separated Butterworth orders to evaluate for the EMG low-pass filter')
    parser.add_argument('--emg-fs-variants', type=str, default='', help='Comma-separated effective sampling rates (Hz) for evaluating low-pass filters')
    parser.add_argument('--emg-bandpass', type=str, default='90,450', help='Band-pass frequency range (low,high) in Hz for EMG')
    parser.add_argument('--emg-bandpass-order', type=int, default=4, help='Butterworth order for the EMG band-pass filter')
    parser.add_argument('--save-dir', type=Path, default=Path('outputs') / 'analysis' / 'process_demonstrations',
                        help='Directory to save generated figure PNGs')
    args = parser.parse_args()

    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        default_root = Path('outputs') / 'logs'
        latest = find_latest_csv(default_root)
        if latest is None:
            raise FileNotFoundError('No CSV provided and no CSV found under outputs/logs')
        print(f'Auto-selected latest CSV: {latest}')
        csv_path = latest
    df = load_trial(csv_path)

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    fs_emg_raw = infer_rate_from_columns(df, 'emg')
    fs_s2 = infer_rate_from_columns(df, 's2')
    fs_s3 = infer_rate_from_columns(df, 's3')
    fs_emg = fs_emg_raw if fs_emg_raw > 0 else (fs_s2 if fs_s2 > 0 else fs_s3 if fs_s3 > 0 else 1000.0)

    force_variants, fs_force = compute_force_variants(df, fs_s2, fs_s3)
    if fs_force <= 0 and fs_s2 > 0:
        fs_force = fs_s2
    elif fs_force <= 0 and fs_s3 > 0:
        fs_force = fs_s3
    if fs_force <= 0:
        fs_force = 100.0

    torque_variants: Dict[str, np.ndarray] = OrderedDict()
    if {'s2_tx', 's2_ty', 's2_tz'}.issubset(df.columns):
        torque_variants['s2_torque'] = median_center(
            df[['s2_tx', 's2_ty', 's2_tz']].to_numpy(dtype=float)
        )
    if {'s3_tx', 's3_ty', 's3_tz'}.issubset(df.columns):
        torque_variants['s3_torque'] = median_center(
            df[['s3_tx', 's3_ty', 's3_tz']].to_numpy(dtype=float)
        )

    emg_raw = df[[f'emg_ch{i}' for i in range(1, 9)]].to_numpy(dtype=float)
    lowpass_orders: List[int] = []
    for token in args.emg_lowpass_orders.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            order_val = int(token)
        except ValueError:
            print(f"[WARN] Ignoring invalid low-pass order '{token}'")
            continue
        if order_val <= 0:
            print(f"[WARN] Ignoring non-positive low-pass order '{token}'")
            continue
        lowpass_orders.append(order_val)

    fs_overrides: List[float] = []
    if args.emg_fs_variants:
        for token in args.emg_fs_variants.split(','):
            token = token.strip()
            if not token:
                continue
            try:
                fs_val = float(token)
            except ValueError:
                print(f"[WARN] Ignoring invalid fs variant '{token}'")
                continue
            if fs_val <= 0:
                print(f"[WARN] Ignoring non-positive fs variant '{token}'")
                continue
            fs_overrides.append(fs_val)

    bandpass_order = args.emg_bandpass_order if args.emg_bandpass_order and args.emg_bandpass_order > 0 else 4

    bandpass_range = (90.0, 450.0)
    if args.emg_bandpass:
        band_tokens = [tok.strip() for tok in args.emg_bandpass.split(',') if tok.strip()]
        if len(band_tokens) == 2:
            try:
                low_val = float(band_tokens[0])
                high_val = float(band_tokens[1])
                if low_val > 0 and high_val > 0 and high_val > low_val:
                    bandpass_range = (low_val, high_val)
                else:
                    print(f"[WARN] Invalid band-pass range '{args.emg_bandpass}', using default {bandpass_range}")
            except ValueError:
                print(f"[WARN] Failed to parse band-pass range '{args.emg_bandpass}', using default {bandpass_range}")
        else:
            print(f"[WARN] Band-pass range '{args.emg_bandpass}' should be 'low,high'; using default {bandpass_range}")

    emg_variants_all = compute_emg_variants(
        emg_raw,
        fs_emg,
        line_freq=args.line_freq,
        envelope_window_sec=args.emg_envelope_window,
        lowpass_cutoff=args.emg_lowpass,
        lowpass_orders=lowpass_orders,
        fs_overrides=fs_overrides,
        bandpass_range=bandpass_range,
        bandpass_order=bandpass_order,
    )

    extra_lowpass_vals = []
    if args.emg_lowpass_multi:
        for token in args.emg_lowpass_multi.split(','):
            token = token.strip()
            if not token:
                continue
            try:
                val = float(token)
            except ValueError:
                print(f"[WARN] Ignoring invalid low-pass cutoff '{token}'")
                continue
            if val <= 0:
                print(f"[WARN] Ignoring non-positive low-pass cutoff '{token}'")
                continue
            extra_lowpass_vals.append(val)

    extra_variant_keys: List[str] = []
    if extra_lowpass_vals:
        for lp_val in extra_lowpass_vals:
            extra_variants = compute_emg_variants(
                emg_raw,
                fs_emg,
                line_freq=args.line_freq,
                envelope_window_sec=args.emg_envelope_window,
                lowpass_cutoff=lp_val,
                lowpass_orders=lowpass_orders,
                fs_overrides=fs_overrides,
                bandpass_range=bandpass_range,
                bandpass_order=bandpass_order,
            )
            suffix = f'_lp{lp_val:g}'
            for key, value in extra_variants.items():
                new_key = f'{key}{suffix}'
                emg_variants_all[new_key] = value
                extra_variant_keys.append(new_key)
            print(f"[INFO] Added EMG variants for low-pass cutoff {lp_val} Hz")
    desired_emg = [
        'raw',
        'raw_centered',
        'bandpass',
        'rectified_bandpass_ma',
        'lowpass',
        'lowpass_smooth',
    ]
    generated_keys = [key for key in emg_variants_all.keys() if key not in desired_emg]
    desired_emg.extend(generated_keys)
    for key in extra_variant_keys:
        if key not in desired_emg:
            desired_emg.append(key)
    emg_variants = OrderedDict((name, emg_variants_all[name]) for name in desired_emg if name in emg_variants_all)

    time = np.arange(len(df), dtype=float)
    if fs_force > 0:
        time = time / fs_force

    time_emg = np.arange(emg_raw.shape[0], dtype=float)
    if fs_emg > 0:
        time_emg = time_emg / fs_emg

    force_subset = OrderedDict((name, force_variants[name]) for name in ['s3_abs'] if name in force_variants)

    variant_results: Dict[str, Dict[str, Dict[str, object]]] = {}

    K_init = 1200.0

    print('=== Mapping Summary per EMG Variant ===')
    for emg_name, emg_proc in emg_variants.items():
        P = emg_proc.T
        variant_results[emg_name] = {}
        print(f"-- EMG Variant: {emg_name} --")
        for force_name, F_mat in force_subset.items():
            F = F_mat.T
            try:
                T_F = compute_tf(P, F)
                H_F = compute_projection_from_tf(T_F)
                T_K, H_K = compute_tk_from_projection(H_F)
                K_raw = T_K @ P + K_init
                residual_F = np.linalg.norm(F - T_F @ P) / max(np.linalg.norm(F), EPS)
                residual_K = np.linalg.norm((np.eye(P.shape[0]) - H_F) - (T_K.T @ np.linalg.inv(T_K @ T_K.T) @ T_K))
                K_smooth = smooth_stiffness(K_raw, fs_force)
                hk_eigs = np.sort(np.linalg.eigvalsh(H_K))[::-1]
                variant_results[emg_name][force_name] = {
                    'T_F': T_F,
                    'H_F': H_F,
                    'T_K': T_K,
                    'H_K_eigs': hk_eigs,
                    'K': K_smooth,
                    'K_raw': K_raw,
                    'residual_F': residual_F,
                    'residual_K': residual_K,
                    'P_samples': emg_proc,
                    'F_samples': F_mat,
                }
                print(f"  Force Variant: {force_name}")
                print(f"    T_F shape: {T_F.shape}, T_K shape: {T_K.shape}")
                print(f"    Force residual: {residual_F:.4e}, Projection residual: {residual_K:.4e}")
                print(f"    H_K eigenvalues (desc): {hk_eigs}")
            except Exception as exc:
                print(f"  Force Variant: {force_name} FAILED -> {exc}")

    successful_variants = {
        emg: [force for force, data in forces.items() if 'K' in data]
        for emg, forces in variant_results.items()
    }

    figures: List[Tuple[str, Figure]] = []
    saved_paths: List[Path] = []

    if not args.no_plot:
        if all(key in emg_variants for key in ['raw', 'bandpass', 'lowpass']):
            raw_data = emg_variants['raw']
            band_data = emg_variants['bandpass']
            low_data = emg_variants['lowpass']
            n_channels = raw_data.shape[1]
            n_cols = 2 if n_channels > 1 else 1
            n_rows = int(np.ceil(n_channels / n_cols))
            fig_core, axes_core = plt.subplots(n_rows, n_cols, figsize=(12, 2.2 * n_rows), sharex=True)
            if isinstance(axes_core, np.ndarray):
                axes_flat = axes_core.flatten()
            else:
                axes_flat = [axes_core]
            channel_labels = [f'ch{idx + 1}' for idx in range(n_channels)]
            for idx, ax in enumerate(axes_flat):
                if idx >= n_channels:
                    ax.axis('off')
                    continue
                ax.plot(time_emg, raw_data[:, idx], label='raw', color='0.5', linewidth=0.7, alpha=0.6)
                ax.plot(time_emg, band_data[:, idx], label='bandpass', color='#1f77b4', linewidth=0.9, alpha=0.9)
                ax.plot(time_emg, low_data[:, idx], label='lowpass', color='#ff7f0e', linewidth=0.9, alpha=0.9)
                ax.set_ylabel(channel_labels[idx])
                ax.grid(True, alpha=0.3)
            axes_flat[-1].set_xlabel('Time (s)')
            axes_flat[0].legend(loc='upper right', fontsize=8)
            fig_core.suptitle('EMG raw vs bandpass vs lowpass (per channel)')
            fig_core.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
            figures.append(('emg_core', fig_core))

        extra_keys = [key for key in emg_variants.keys() if key not in {'raw', 'raw_centered', 'bandpass', 'lowpass'}]
        if extra_keys:
            fig_emg, axes_emg = plt.subplots(len(extra_keys), 1, figsize=(12, 2.4 * len(extra_keys)), sharex=True)
            if len(extra_keys) == 1:
                axes_emg = [axes_emg]
            for ax, key in zip(axes_emg, extra_keys):
                emg_proc = emg_variants[key]
                for ch in range(emg_proc.shape[1]):
                    ax.plot(time_emg, emg_proc[:, ch], linewidth=0.8, alpha=0.6)
                ax.set_ylabel(key)
                ax.grid(True, alpha=0.3)
            axes_emg[-1].set_xlabel('Time (s)')
            fig_emg.suptitle('EMG additional variants')
            fig_emg.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
            figures.append(('emg_variants', fig_emg))

        # Frequency-domain comparison for key variants (channel 1)
        freq_variants = [name for name in emg_variants if name.startswith('lowpass') or name.startswith('bandpass')]
        if freq_variants and fs_emg > 0:
            fig_fft, ax_fft = plt.subplots(figsize=(12, 4))
            freqs = np.fft.rfftfreq(emg_raw.shape[0], d=1.0 / fs_emg)
            for name in freq_variants:
                data = emg_variants[name][:, 0]
                spectrum = np.abs(np.fft.rfft(data - np.mean(data)))
                ax_fft.plot(freqs, spectrum, label=name)
            ax_fft.set_xlim(0.0, min(500.0, freqs[-1]))
            ax_fft.set_xlabel('Frequency (Hz)')
            ax_fft.set_ylabel('Magnitude (|FFT|)')
            ax_fft.set_title('EMG Frequency Domain (channel 1)')
            ax_fft.grid(True, alpha=0.3)
            ax_fft.legend(loc='upper right')
            fig_fft.tight_layout()
            figures.append(('emg_fft', fig_fft))

        for emg_idx, (emg_name, force_dict) in enumerate(variant_results.items()):
            successful_force = successful_variants.get(emg_name, [])
            if not successful_force:
                continue
            fig_k, axes_k = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
            axis_labels = ['x', 'y', 'z']
            colormap = plt.get_cmap('tab10')
            colors = colormap(np.linspace(0, 1, len(successful_force)))
            for idx_axis, axis_label in enumerate(axis_labels):
                ax = axes_k[idx_axis]
                for color, force_name in zip(colors, successful_force):
                    K = np.asarray(force_dict[force_name]['K'])
                    ax.plot(time, K[idx_axis], label=force_name, color=color, linewidth=1.5)
                ax.set_ylabel(f'K_{axis_label}')
                ax.grid(True, alpha=0.3)
            axes_k[-1].set_xlabel('Time (s)')
            axes_k[0].legend(loc='upper right')
            fig_k.suptitle(f'Stiffness: EMG={emg_name} (K_init = 1200)')
            fig_k.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
            safe_name = emg_name.replace('/', '_')
            figures.append((f'stiffness_{safe_name}', fig_k))

        fig_force, axes_force = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        axis_labels = ['x', 'y', 'z']
        s2_signed = force_variants.get('s2_signed')
        s2_abs = force_variants.get('s2_abs')
        s3_signed = force_variants.get('s3_signed')
        s3_abs = force_variants.get('s3_abs')
        s2_torque = torque_variants.get('s2_torque')
        s3_torque = torque_variants.get('s3_torque')
        for idx_axis, axis_label in enumerate(axis_labels):
            axf = axes_force[idx_axis]
            if s2_signed is not None:
                axf.plot(time, s2_signed[:, idx_axis], label=f's2_F{axis_label}', color='#1f77b4', linewidth=1.0)
            if s2_abs is not None:
                axf.plot(time, s2_abs[:, idx_axis], label=f's2_|F{axis_label}|', color='#1f77b4', linestyle='--', linewidth=1.2)
            if s3_signed is not None:
                axf.plot(time, s3_signed[:, idx_axis], label=f's3_F{axis_label}', color='#ff7f0e', linewidth=1.0)
            if s3_abs is not None:
                axf.plot(time, s3_abs[:, idx_axis], label=f's3_|F{axis_label}|', color='#ff7f0e', linestyle='--', linewidth=1.2)
            if s2_torque is not None:
                axf.plot(time, s2_torque[:, idx_axis], label=f's2_T{axis_label}', color='#2ca02c', linestyle='-.', linewidth=1.0)
            if s3_torque is not None:
                axf.plot(time, s3_torque[:, idx_axis], label=f's3_T{axis_label}', color='#d62728', linestyle=':', linewidth=1.0)
            axf.set_ylabel(f'Axis {axis_label.upper()}')
            axf.grid(True, alpha=0.3)
            axf.legend(loc='upper right', ncol=2)
        axes_force[-1].set_xlabel('Time (s)')
        fig_force.suptitle('Force/Torque Components (sensor-wise)')
        fig_force.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        figures.append(('forces', fig_force))

    if figures:
        for label, fig in figures:
            out_path = save_dir / f'{csv_path.stem}_{label}.png'
            fig.savefig(out_path, dpi=160)
            saved_paths.append(out_path)

        plt.show()
        for _, fig in figures:
            plt.close(fig)

        if saved_paths:
            print('Saved figures:')
            for path in saved_paths:
                print(f'  {path}')


if __name__ == '__main__':
    main()
