#!/usr/bin/env python3
"""Train and evaluate multiple generative policies for stiffness prediction.

This utility complements ``diffusion_policy_full.py`` by exposing additional
baselines so you can compare modelling choices under a common CLI.

Models currently supported
--------------------------
- ``diffusion``  : Transformer-based diffusion policy (imports existing module)
- ``gaussian``   : Conditional Gaussian policy with learned diagonal covariance
- ``cvae``       : Conditional variational auto-encoder with latent sampling

Example usage
-------------
Train three models (diffusion / gaussian / cvae):
    python3 scripts/generative_model_suite.py train --model diffusion \
        --data-dir outputs/logs/20250929 --epochs 50
    python3 scripts/generative_model_suite.py train --model gaussian  \
        --data-dir outputs/logs/20250929 --epochs 50
    python3 scripts/generative_model_suite.py train --model cvae      \
        --data-dir outputs/logs/20250929 --epochs 50 --latent-dim 64

Sample from a trained checkpoint:
    python3 scripts/generative_model_suite.py sample \
        --checkpoint outputs/models/cvae/best_model.pth \
        --csv outputs/logs/20250929/trial_1.csv
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == "__main__":
    import sys

    scripts_dir = Path(__file__).parent
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))

from diffusion_policy_full import (  # type: ignore
    DEFAULT_FORCE_VARIANT,
    DiffusionConfig,
    DiffusionPolicy,
    DemonstrationDataset,
    compute_force_variants,
    infer_rate_from_columns,
    load_trial,
    split_datasets,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Diagonal Gaussian negative log-likelihood summed per sample."""

    log_std = torch.clamp(log_std, min=-8.0, max=4.0)
    inv_var = torch.exp(-2.0 * log_std)
    mse_term = (target - mean) ** 2 * inv_var
    nll = 0.5 * (mse_term + 2.0 * log_std + math.log(2.0 * math.pi))
    return nll.sum(dim=-1)


def flatten_actions(actions: torch.Tensor) -> torch.Tensor:
    return actions.reshape(actions.size(0), -1)


# ---------------------------------------------------------------------------
# Conditional Gaussian baseline
# ---------------------------------------------------------------------------


class ConditionalGaussianPolicy(nn.Module):
    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden_dim
        output_dim = cfg.action_dim * cfg.horizon

        self.backbone = nn.Sequential(
            nn.Linear(cfg.obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, output_dim)
        self.log_std_head = nn.Linear(hidden, output_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        return mean, log_std

    def compute_loss(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, log_std = self(obs)
        target = flatten_actions(actions)
        nll = gaussian_nll(target, mean, log_std)
        return nll.mean()

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self(obs)
        if deterministic:
            draw = mean
        else:
            std = torch.exp(torch.clamp(log_std, min=-8.0, max=4.0))
            draw = mean + std * torch.randn_like(std)
        return draw.reshape(-1, self.cfg.horizon, self.cfg.action_dim)


# ---------------------------------------------------------------------------
# Conditional VAE baseline
# ---------------------------------------------------------------------------


class ConditionalVAEPolicy(nn.Module):
    def __init__(self, cfg: DiffusionConfig, latent_dim: int = 32) -> None:
        super().__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim
        hidden = cfg.hidden_dim
        flat_action = cfg.action_dim * cfg.horizon

        encoder_inputs = cfg.obs_dim + flat_action
        decoder_inputs = cfg.obs_dim + latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(encoder_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden, latent_dim)
        self.enc_logvar = nn.Linear(hidden, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(decoder_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.dec_mean = nn.Linear(hidden, flat_action)
        self.dec_log_std = nn.Linear(hidden, flat_action)

    def encode(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, flatten_actions(actions)], dim=-1)
        h = self.encoder(x)
        mu = self.enc_mu(h)
        logvar = torch.clamp(self.enc_logvar(h), min=-8.0, max=8.0)
        return mu, logvar

    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, z], dim=-1)
        h = self.decoder(x)
        mean = self.dec_mean(h)
        log_std = self.dec_log_std(h)
        return mean, log_std

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_loss(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(obs, actions)
        z = self.reparameterize(mu, logvar)
        mean, log_std = self.decode(obs, z)

        target = flatten_actions(actions)
        recon = gaussian_nll(target, mean, log_std)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return (recon + kl).mean()

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        batch = obs.size(0)
        if deterministic:
            z = torch.zeros(batch, self.latent_dim, device=obs.device)
        else:
            z = torch.randn(batch, self.latent_dim, device=obs.device)
        mean, log_std = self.decode(obs, z)
        if deterministic:
            draw = mean
        else:
            std = torch.exp(torch.clamp(log_std, min=-8.0, max=4.0))
            draw = mean + std * torch.randn_like(std)
        return draw.reshape(-1, self.cfg.horizon, self.cfg.action_dim)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def build_model(model_name: str, cfg: DiffusionConfig, latent_dim: int, device: torch.device) -> nn.Module:
    name = model_name.lower()
    if name == "diffusion":
        return DiffusionPolicy(cfg).to(device)
    if name == "gaussian":
        return ConditionalGaussianPolicy(cfg).to(device)
    if name == "cvae":
        return ConditionalVAEPolicy(cfg, latent_dim=latent_dim).to(device)
    raise ValueError(f"Unsupported model '{model_name}'")


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    running = 0.0
    for obs, actions in tqdm(loader, leave=False):
        obs = obs.to(device)
        actions = actions.to(device)
        loss = model.compute_loss(obs, actions)  # type: ignore[attr-defined]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item()
    return running / max(len(loader), 1)


def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for obs, actions in loader:
            obs = obs.to(device)
            actions = actions.to(device)
            loss = model.compute_loss(obs, actions)  # type: ignore[attr-defined]
            total += loss.item()
    return total / max(len(loader), 1)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    cfg = DiffusionConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        horizon=args.horizon,
        n_diffusion_steps=args.steps,
        hidden_dim=args.hidden_dim,
    )

    train_loader, val_loader = split_datasets(Path(args.data_dir), cfg)
    sample_obs, sample_actions = train_loader.dataset[0]
    cfg.obs_dim = int(sample_obs.shape[-1])
    cfg.action_dim = int(sample_actions.shape[-1])

    model = build_model(args.model, cfg, args.latent_dim, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_root = Path(args.output_dir) / args.model.lower()
    out_root.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, object] = {"model": args.model, "latent_dim": args.latent_dim}

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        scheduler.step()

        print(
            f"[{args.model}] Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}"
        )

        ckpt = {
            "model_type": args.model,
            "config": asdict(cfg),
            "epoch": epoch,
            "val_loss": val_loss,
            "state_dict": model.state_dict(),
            "meta": meta,
        }

        torch.save(ckpt, out_root / f"latest_epoch_{epoch:03d}.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_root / "best_model.pth")
            print(f"  -> Best checkpoint updated (val {val_loss:.4f})")

    with open(out_root / "config.json", "w") as f:
        json.dump({"config": asdict(cfg), "meta": meta}, f, indent=2)

    print(f"Training complete. Best validation loss: {best_val:.4f}")


def build_observation(csv_path: Optional[Path], cfg: DiffusionConfig) -> np.ndarray:
    if csv_path is None:
        return np.zeros(cfg.obs_dim, dtype=np.float32)

    df = load_trial(csv_path)
    force_variants, _ = compute_force_variants(
        df,
        infer_rate_from_columns(df, "s2"),
        infer_rate_from_columns(df, "s3"),
    )
    forces = force_variants.get(DEFAULT_FORCE_VARIANT)
    if forces is None:
        raise ValueError(f"{DEFAULT_FORCE_VARIANT} force variant missing in provided CSV")

    ee_positions = df[["ee_px", "ee_py", "ee_pz"]].to_numpy(dtype=float)
    eccentricity = df["deform_ecc"].to_numpy(dtype=float)

    position = np.nan_to_num(ee_positions[0], nan=0.0)
    force = np.nan_to_num(forces[0], nan=0.0)
    ecc_val = float(np.nan_to_num(eccentricity[0], nan=0.0))
    obs_vec = np.concatenate([position, force, [ecc_val]]).astype(np.float32)

    if obs_vec.shape[0] != cfg.obs_dim:
        if obs_vec.shape[0] > cfg.obs_dim:
            obs_vec = obs_vec[: cfg.obs_dim]
        else:
            obs_vec = np.pad(obs_vec, (0, cfg.obs_dim - obs_vec.shape[0]))
    return obs_vec


def cmd_sample(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ckpt = torch.load(Path(args.checkpoint), map_location=device)
    cfg = DiffusionConfig(**ckpt["config"])
    model_type = ckpt.get("model_type", "diffusion")
    latent_dim = ckpt.get("meta", {}).get("latent_dim", args.latent_dim)

    model = build_model(model_type, cfg, latent_dim, device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    obs_vec = build_observation(Path(args.csv) if args.csv else None, cfg)
    obs_tensor = torch.from_numpy(obs_vec).unsqueeze(0).to(device)

    traj = model.sample(obs_tensor, deterministic=args.deterministic)[0].cpu().numpy()

    print(f"Generated trajectory using {model_type} (first 5 steps):")
    for i, triple in enumerate(traj[:5]):
        print(f"  step {i:02d}: K = [{triple[0]:.2f}, {triple[1]:.2f}, {triple[2]:.2f}]")

    if args.output:
        Path(args.output).write_text(json.dumps(traj.tolist(), indent=2))
        print(f"Full trajectory saved to {args.output}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or sample multiple generative policies")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a generative policy")
    train.add_argument("--model", type=str, choices=["diffusion", "gaussian", "cvae"], required=True)
    train.add_argument("--data-dir", type=str, required=True)
    train.add_argument("--output-dir", type=str, default="outputs/models")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--horizon", type=int, default=16)
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--latent-dim", type=int, default=32, help="Latent size for CVAE (ignored otherwise)")
    train.set_defaults(func=cmd_train)

    sample = subparsers.add_parser("sample", help="Sample from a trained policy")
    sample.add_argument("--checkpoint", type=str, required=True)
    sample.add_argument("--device", type=str, default="cpu")
    sample.add_argument("--csv", type=str, default="")
    sample.add_argument("--latent-dim", type=int, default=32)
    sample.add_argument("--deterministic", action="store_true")
    sample.add_argument("--output", type=str, default="")
    sample.set_defaults(func=cmd_sample)

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
