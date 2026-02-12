#!/usr/bin/env python3
"""End-to-end Diffusion Policy implementation for impedance imitation learning.

This script bundles:
    • Dataset preparation from trial CSVs (ee pose + force + deformity → stiffness trajectories)
  • Transformer-based diffusion model definition (denoising network + sampler)
  • Training loop with validation + checkpointing
  • Simple sampling utility to generate stiffness plans from a CSV trial

Example usage
--------------
Train a model (saves under outputs/models/diffusion_policy):
    python3 scripts/diffusion_policy_full.py train \
        --data-dir outputs/logs/20250929 \
        --epochs 50 --device cuda

Sample a trajectory for sanity-check (printed to stdout / saved JSON):
    python3 scripts/diffusion_policy_full.py sample \
        --model-path outputs/models/diffusion_policy/best_model.pth \
        --csv outputs/logs/20250929/trial_1.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow relative imports from scripts directory
if __name__ == "__main__":
    import sys
    scripts_dir = Path(__file__).parent
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))

from process_demonstrations import (  # type: ignore
    compute_emg_variants,
    compute_force_variants,
    infer_rate_from_columns,
    load_trial,
)


DEFAULT_FORCE_VARIANT = "s3_signed"

# ---------------------------------------------------------------------------
# Configuration and helper utilities
# ---------------------------------------------------------------------------


@dataclass
class DiffusionConfig:
    """Hyper-parameters for Diffusion Policy training."""

    obs_dim: int = 7  # End-effector position(3) + external force(3) + eccentricity(1)
    action_dim: int = 3  # Stiffness Kx, Ky, Kz
    horizon: int = 16  # Number of future steps to predict per sample
    n_diffusion_steps: int = 100
    hidden_dim: int = 256
    n_heads: int = 8
    n_layers: int = 6
    beta_start: float = 1e-4
    beta_end: float = 0.02
    batch_size: int = 64
    learning_rate: float = 1e-4
    ema_decay: float = 0.995  # Optional exponential moving average of weights


# ---------------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------------


class DemonstrationDataset(Dataset):
    """Sliding-window dataset built from multiple trial CSVs."""

    def __init__(
        self,
        csv_paths: List[Path],
        horizon: int = 16,
        stride: int = 4,
        emg_variant: str = "lowpass_smooth",
        force_variant: str = DEFAULT_FORCE_VARIANT,
    ) -> None:
        self.horizon = horizon
        self.stride = max(1, stride)
        self.samples: List[Dict[str, np.ndarray]] = []

        for csv_path in csv_paths:
            try:
                df = load_trial(csv_path)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[WARN] Failed to load {csv_path}: {exc}")
                continue

            # EMG processing -------------------------------------------------
            emg_raw = df[[f"emg_ch{i}" for i in range(1, 9)]].to_numpy(dtype=float)
            fs_emg = infer_rate_from_columns(df, "emg")
            if fs_emg <= 0:
                fs_emg = 100.0  # fallback

            emg_variants = compute_emg_variants(
                emg_raw,
                fs_emg,
                line_freq=60.0,
                envelope_window_sec=0.15,
                lowpass_cutoff=10.0,
            )
            if emg_variant not in emg_variants:
                print(f"[WARN] EMG variant '{emg_variant}' missing in {csv_path.name}")
                continue
            emg_processed = emg_variants[emg_variant]

            # Force processing -----------------------------------------------
            fs_s2 = infer_rate_from_columns(df, "s2")
            fs_s3 = infer_rate_from_columns(df, "s3")
            force_variants, _ = compute_force_variants(df, fs_s2, fs_s3)
            if force_variant not in force_variants:
                print(
                    f"[WARN] Force variant '{force_variant}' missing in {csv_path.name}"
                )
                continue
            forces = force_variants[force_variant]

            ee_positions = df[["ee_px", "ee_py", "ee_pz"]].to_numpy(dtype=float)
            eccentricity = df["deform_ecc"].to_numpy(dtype=float)

            # Simple stiffness proxy based on EMG intensity (placeholder)
            emg_power = np.mean(np.square(emg_processed), axis=1)
            emg_power_norm = (emg_power - np.min(emg_power)) / (
                np.max(emg_power) - np.min(emg_power) + 1e-8
            )
            stiffness = 1200.0 + 600.0 * emg_power_norm[:, None] * np.ones((1, 3))

            # Construct observation/action pairs via sliding window ----------
            for start in range(0, len(df) - horizon, self.stride):
                end = start + horizon

                force_t = np.nan_to_num(forces[start], nan=0.0)
                position_t = np.nan_to_num(ee_positions[start], nan=0.0)
                ecc_t = float(np.nan_to_num(eccentricity[start], nan=0.0))
                obs = np.concatenate([position_t, force_t, [ecc_t]]).astype(
                    np.float32
                )

                action_seq = stiffness[start:end].astype(np.float32)
                self.samples.append({"obs": obs, "actions": action_seq})

        if not self.samples:
            raise RuntimeError("No valid samples constructed from provided CSVs")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        return torch.from_numpy(item["obs"]), torch.from_numpy(item["actions"])


def split_datasets(
    root_dir: Path, config: DiffusionConfig, train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """Create train/validation splits and wrap them in DataLoaders."""

    csv_paths = sorted(Path(root_dir).glob("trial_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No trial_*.csv files under {root_dir}")

    split = max(1, int(len(csv_paths) * train_ratio))
    train_csvs = csv_paths[:split]
    val_csvs = csv_paths[split:] or csv_paths[-1:]

    train_dataset = DemonstrationDataset(
        train_csvs, horizon=config.horizon, stride=4
    )
    val_dataset = DemonstrationDataset(val_csvs, horizon=config.horizon, stride=8)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Diffusion model components
# ---------------------------------------------------------------------------


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(
                np.log(1e-4), np.log(1e4), steps=half, device=timesteps.device
            )
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0) 
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TransformerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + out)
        if context is not None:
            out, _ = self.cross_attn(x, context, context)
            x = self.norm2(x + out)
        out = self.ffn(x)
        return self.norm3(x + out)


class DiffusionTransformer(nn.Module):
    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.obs_proj = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.action_proj = nn.Linear(
            config.action_dim * config.horizon, config.hidden_dim
        )
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(config.hidden_dim, config.n_heads) for _ in range(config.n_layers)]
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.action_dim * config.horizon),
        )

    def forward(
        self, noisy_actions: torch.Tensor, obs: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        obs_emb = self.obs_proj(obs)[:, None, :]
        act_emb = self.action_proj(noisy_actions)[:, None, :]
        time_emb = self.time_embed(timesteps)[:, None, :]

        x = act_emb + time_emb
        for block in self.blocks:
            x = block(x, context=obs_emb)
        return self.out_proj(x.squeeze(1))


class DiffusionPolicy(nn.Module):
    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.cfg = config
        self.denoiser = DiffusionTransformer(config)

        betas = torch.linspace(config.beta_start, config.beta_end, config.n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([
            torch.ones(1), alphas_cumprod[:-1]
        ])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    def add_noise(
        self, actions: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_alpha * actions + sqrt_one_minus * noise

    def compute_loss(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        bsz = obs.size(0)
        horizon = self.cfg.horizon
        actions_flat = actions.reshape(bsz, horizon * self.cfg.action_dim)
        t = torch.randint(0, self.cfg.n_diffusion_steps, (bsz,), device=obs.device)
        noise = torch.randn_like(actions_flat)
        noisy_actions = self.add_noise(actions_flat, noise, t)
        pred_noise = self.denoiser(noisy_actions, obs, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        self.eval()
        bsz = obs.size(0)
        device = obs.device
        horizon = self.cfg.horizon
        act_dim = self.cfg.action_dim
        latents = torch.randn(bsz, horizon * act_dim, device=device)

        for step in reversed(range(self.cfg.n_diffusion_steps)):
            t = torch.full((bsz,), step, device=device, dtype=torch.long)
            epsilon = self.denoiser(latents, obs, t)

            alpha = self.alphas[step]
            alpha_bar = self.alphas_cumprod[step]
            alpha_bar_prev = self.alphas_cumprod_prev[step]

            x0_pred = (latents - torch.sqrt(1 - alpha_bar) * epsilon) / torch.sqrt(
                alpha_bar
            )

            if deterministic:
                latents = (
                    torch.sqrt(alpha_bar_prev) * x0_pred
                    + torch.sqrt(1 - alpha_bar_prev) * epsilon
                )
            else:
                noise = torch.randn_like(latents) if step > 0 else 0.0
                latents = (
                    torch.sqrt(alpha) * latents
                    + torch.sqrt(1 - alpha) * epsilon
                )
                latents += torch.sqrt(self.betas[step]) * noise

        return latents.reshape(bsz, horizon, act_dim)


# ---------------------------------------------------------------------------
# Training / evaluation routines
# ---------------------------------------------------------------------------


def train_epoch(
    model: DiffusionPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running = 0.0
    for obs, actions in tqdm(loader, desc=f"Epoch {epoch}"):
        obs = obs.to(device)
        actions = actions.to(device)
        loss = model.compute_loss(obs, actions)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item()
    return running / len(loader)


def eval_epoch(model: DiffusionPolicy, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for obs, actions in loader:
            obs = obs.to(device)
            actions = actions.to(device)
            total += model.compute_loss(obs, actions).item()
    return total / len(loader)


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    cfg = DiffusionConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        horizon=args.horizon,
        n_diffusion_steps=args.steps,
        hidden_dim=args.hidden_dim,
    )

    device = torch.device(args.device)

    train_loader, val_loader = split_datasets(Path(args.data_dir), cfg)

    sample_obs, sample_actions = train_loader.dataset[0]
    cfg.obs_dim = int(sample_obs.shape[-1])
    cfg.action_dim = int(sample_actions.shape[-1])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    model = DiffusionPolicy(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = eval_epoch(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                out_dir / "best_model.pth",
            )
            print(f"  -> Best model updated (val {val_loss:.4f})")

        if epoch % 10 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                out_dir / f"checkpoint_epoch_{epoch:03d}.pth",
            )

    print("Training complete. Best val loss:", best_val)


def cmd_sample(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.model_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = DiffusionConfig(**checkpoint["config"])
    device = torch.device(args.device)

    model = DiffusionPolicy(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Build a single observation either from CSV or zeros
    if args.csv:
        df = load_trial(Path(args.csv))
        force_variants, _ = compute_force_variants(
            df,
            infer_rate_from_columns(df, "s2"),
            infer_rate_from_columns(df, "s3"),
        )
        forces = force_variants.get(DEFAULT_FORCE_VARIANT)
        if forces is None:
            raise ValueError(
                f"{DEFAULT_FORCE_VARIANT} force variant missing in provided CSV"
            )

        ee_positions = df[["ee_px", "ee_py", "ee_pz"]].to_numpy(dtype=float)
        eccentricity = df["deform_ecc"].to_numpy(dtype=float)

        position = np.nan_to_num(ee_positions[0], nan=0.0)
        force = np.nan_to_num(forces[0], nan=0.0)
        ecc_val = float(np.nan_to_num(eccentricity[0], nan=0.0))
        obs_vec = np.concatenate([position, force, [ecc_val]]).astype(np.float32)
    else:
        obs_vec = np.zeros(cfg.obs_dim, dtype=np.float32)

    if obs_vec.shape[0] != cfg.obs_dim:
        if obs_vec.shape[0] > cfg.obs_dim:
            obs_vec = obs_vec[: cfg.obs_dim]
        else:
            obs_vec = np.pad(obs_vec, (0, cfg.obs_dim - obs_vec.shape[0]))

    obs_tensor = torch.from_numpy(obs_vec).unsqueeze(0).to(device)
    traj = model.sample(obs_tensor, deterministic=True)[0].cpu().numpy()

    print("Generated stiffness trajectory (first 5 steps):")
    for i, triple in enumerate(traj[:5]):
        print(f"  step {i:02d}: K = [{triple[0]:.1f}, {triple[1]:.1f}, {triple[2]:.1f}]")

    if args.output:
        Path(args.output).write_text(json.dumps(traj.tolist(), indent=2))
        print(f"Full trajectory saved to {args.output}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diffusion Policy toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train diffusion policy")
    train.add_argument("--data-dir", type=str, required=True)
    train.add_argument("--output-dir", type=str, default="outputs/models/diffusion_policy")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--horizon", type=int, default=16)
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.set_defaults(func=cmd_train)

    sample = subparsers.add_parser("sample", help="Sample stiffness trajectory")
    sample.add_argument("--model-path", type=str, required=True)
    sample.add_argument("--device", type=str, default="cpu")
    sample.add_argument("--csv", type=str, default="")
    sample.add_argument("--output", type=str, default="")
    sample.set_defaults(func=cmd_sample)

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
