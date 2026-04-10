"""
Training script for the hybrid THGNN × MaGNet model.

Loss function: composite MSE + Spearman IC + dispersion penalty (from THGNN).
Model:         HybridStockModel (MAGE temporal encoder + pos/neg GAT + GPH).

Usage examples
--------------
# Train on existing THGNN Nifty50 data (default paths):
    python train_hybrid.py

# Custom date range:
    python train_hybrid.py --train-start-date 2018-01-01 --train-end-date 2023-12-31 \
        --test-start-date 2024-01-01 --test-end-date 2024-12-31

# Larger embed dim for N=300:
    python train_hybrid.py --embed-dim 128 --num-hyper-edges 64

# Custom data directory:
    python train_hybrid.py --data-dir /path/to/data_train_predict
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import AllGraphDataSampler
from model.hybrid_model import HybridStockModel


# Default paths: point at the THGNN sibling directory's data
_HERE = Path(__file__).resolve().parent
_THGNN_DIR = _HERE.parent / "THGNN"
DATA_DIR = _THGNN_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "data_train_predict"
MODEL_DIR = DATA_DIR / "model_saved"
PLOT_DIR = DATA_DIR / "plots"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

@dataclass
class SplitIndices:
    train_start: int
    train_end_exclusive: int
    test_start: int
    test_end_exclusive: int
    pre_data: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train hybrid THGNN×MaGNet with MSE + IC + dispersion loss."
    )
    # Data / paths
    p.add_argument("--data-dir", type=Path, default=TRAIN_DATA_DIR)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    # Date splits
    p.add_argument("--train-start-date", type=str, default="2015-01-01")
    p.add_argument("--train-end-date",   type=str, default="2024-12-31")
    p.add_argument("--test-start-date",  type=str, default="2025-01-01")
    p.add_argument("--test-end-date",    type=str, default="2026-12-31")
    # Training
    p.add_argument("--epochs",         type=int,   default=80)
    p.add_argument("--lr",             type=float, default=5e-4)
    p.add_argument("--weight-decay",   type=float, default=5e-5)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--patience",       type=int,   default=10)
    p.add_argument("--seed",           type=int,   default=42)
    # Model architecture
    p.add_argument("--in-features",      type=int, default=12,
                   help="Input feature dimension per stock per timestep.")
    p.add_argument("--embed-dim",        type=int, default=64,
                   help="Core embedding dimension D used throughout the model.")
    p.add_argument("--num-mage-layers",  type=int, default=1,
                   help="Number of stacked MAGE blocks.")
    p.add_argument("--num-moe-experts",  type=int, default=4,
                   help="Number of experts in each SparseMoE layer.")
    p.add_argument("--num-mha-heads",    type=int, default=2,
                   help="Number of heads in MAGE's temporal self-attention.")
    p.add_argument("--gat-heads",        type=int, default=8,
                   help="Number of heads in pos/neg GAT layers.")
    p.add_argument("--gat-out-features", type=int, default=8,
                   help="Per-head output dimension in pos/neg GAT.")
    p.add_argument("--num-hyper-edges",  type=int, default=32,
                   help="Number of hyperedges M in the GPH module. "
                        "Increase for larger universes (e.g. 64 for N=300).")
    # Loss weights
    p.add_argument("--mse-weight",          type=float, default=0.7)
    p.add_argument("--ic-weight",           type=float, default=0.5)
    p.add_argument("--dispersion-weight",   type=float, default=0.3)
    p.add_argument("--min-dispersion-ratio", type=float, default=0.5)
    p.add_argument("--max-dispersion-ratio", type=float, default=2.0)
    p.add_argument("--return-scale", type=float, default=0.02,
                   help="Typical daily return std used to normalise MSE.")
    p.add_argument("--target-horizon", type=int, default=0,
                   help="Which label horizon to train on (0=next-day, 1, 2).")
    p.add_argument("--ic-warmup-epochs", type=int, default=3,
                   help="Ramp IC weight from 0 to --ic-weight over this many epochs.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _index_for_date(file_dates: list[pd.Timestamp], date_str: str) -> int:
    dt = pd.to_datetime(date_str).normalize()
    for idx, d in enumerate(file_dates):
        if d >= dt:
            return idx
    return len(file_dates) - 1


def _exclusive_end_index_for_date(file_dates: list[pd.Timestamp], date_str: str) -> int:
    dt = pd.to_datetime(date_str).normalize()
    for idx, d in enumerate(file_dates):
        if d > dt:
            return idx
    return len(file_dates)


def compute_split_indices(files: list[str], args: argparse.Namespace) -> SplitIndices:
    file_dates = [pd.to_datetime(Path(name).stem).normalize() for name in files]
    min_date, max_date = file_dates[0], file_dates[-1]

    train_start        = _index_for_date(file_dates, args.train_start_date)
    train_end_exclusive = _exclusive_end_index_for_date(file_dates, args.train_end_date)
    test_start         = _index_for_date(file_dates, args.test_start_date)
    test_end_exclusive  = _exclusive_end_index_for_date(file_dates, args.test_end_date)

    train_end_exclusive = min(train_end_exclusive, len(files))
    test_end_exclusive  = min(test_end_exclusive, len(files))

    if not (train_start < train_end_exclusive <= test_start < test_end_exclusive):
        raise ValueError(
            "Invalid split ordering. "
            f"Resolved indices: train [{train_start}, {train_end_exclusive}), "
            f"test [{test_start}, {test_end_exclusive})."
        )
    if train_end_exclusive <= train_start:
        raise ValueError("Training split is empty.")
    if test_end_exclusive <= test_start:
        raise ValueError("Test split is empty.")

    if min_date > pd.to_datetime(args.train_start_date):
        print(f"Warning: train start {args.train_start_date} earlier than data; "
              f"using {min_date.date()}.")
    if max_date < pd.to_datetime(args.test_end_date):
        print(f"Warning: test end {args.test_end_date} beyond data; "
              f"using {max_date.date()}.")

    return SplitIndices(
        train_start=train_start,
        train_end_exclusive=train_end_exclusive,
        test_start=test_start,
        test_end_exclusive=test_end_exclusive,
        pre_data=Path(files[train_end_exclusive - 1]).stem,
    )


def extract_data(data_dict: dict, device: str):
    """Move a sample dict to device and strip spurious batch dimensions."""
    pos_adj  = data_dict["pos_adj"].to(device)
    neg_adj  = data_dict["neg_adj"].to(device)
    features = data_dict["features"].to(device)
    labels   = data_dict["labels"].to(device)

    def _safe_squeeze(t: torch.Tensor, min_dims: int) -> torch.Tensor:
        while t.dim() > min_dims and t.size(0) == 1:
            t = t.squeeze(0)
        return t

    features = _safe_squeeze(features, 3)   # keep (N, T, F)
    pos_adj  = _safe_squeeze(pos_adj, 2)    # keep (N, N)
    neg_adj  = _safe_squeeze(neg_adj, 2)
    if labels.dim() > 1 and labels.size(-1) == 1:
        labels = labels.squeeze(-1)

    mask = data_dict["mask"]
    return pos_adj, neg_adj, features, labels, mask


def _mask_to_tensor(mask, length: int, device: torch.device) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        return mask.to(device).bool()
    return torch.tensor(mask, device=device, dtype=torch.bool)


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(-1) if x.dim() == 1 else x


# ---------------------------------------------------------------------------
# Loss functions  (identical to THGNN train_ic_ranked.py)
# ---------------------------------------------------------------------------

def _soft_rank(x: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    diff = x.unsqueeze(0) - x.unsqueeze(1)
    return torch.sigmoid(diff / temperature).sum(dim=1)


def cross_sectional_ic(pred: torch.Tensor, target: torch.Tensor,
                       eps: float = 1e-8, temperature: float = 0.05) -> torch.Tensor:
    pred_flat, target_flat = pred.reshape(-1), target.reshape(-1)
    if pred_flat.numel() < 2:
        return torch.tensor(0.0, device=pred.device)
    pred_rank   = _soft_rank(pred_flat, temperature)
    target_rank = _soft_rank(target_flat.detach(), temperature).detach()
    pred_r   = pred_rank   - pred_rank.mean()
    target_r = target_rank - target_rank.mean()
    denom = torch.sqrt((pred_r ** 2).sum() * (target_r ** 2).sum() + eps)
    if denom.item() <= eps:
        return torch.tensor(0.0, device=pred.device)
    return torch.sum(pred_r * target_r) / denom


def _exact_ranks(x: torch.Tensor) -> torch.Tensor:
    x_flat = x.reshape(-1)
    order  = torch.argsort(x_flat)
    ranks  = torch.empty_like(x_flat)
    ranks[order] = torch.arange(1, x_flat.numel() + 1,
                                device=x.device, dtype=torch.float32)
    return ranks


def exact_spearman_ic(pred: torch.Tensor, target: torch.Tensor,
                      eps: float = 1e-8) -> torch.Tensor:
    pred_flat, target_flat = pred.reshape(-1), target.reshape(-1)
    if pred_flat.numel() < 2:
        return torch.tensor(0.0, device=pred.device)
    pred_rank   = _exact_ranks(pred_flat)
    target_rank = _exact_ranks(target_flat)
    pred_r   = pred_rank   - pred_rank.mean()
    target_r = target_rank - target_rank.mean()
    denom = torch.sqrt((pred_r ** 2).sum() * (target_r ** 2).sum() + eps)
    if denom.item() <= eps:
        return torch.tensor(0.0, device=pred.device)
    return torch.sum(pred_r * target_r) / denom


def composite_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mse_weight: float,
    ic_weight: float,
    dispersion_weight: float,
    min_dispersion_ratio: float,
    max_dispersion_ratio: float = 2.0,
    temperature: float = 0.05,
    return_scale: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred   = _ensure_2d(pred)
    target = _ensure_2d(target)

    pred_flat   = pred.reshape(-1)
    target_flat = target.reshape(-1)
    pred_cs   = pred_flat   - pred_flat.mean()
    target_cs = target_flat - target_flat.detach().mean()

    mse       = F.mse_loss(pred_cs, target_cs) / (return_scale ** 2)
    soft_corr = cross_sectional_ic(pred, target, temperature=temperature)
    ic_loss   = 1.0 - soft_corr

    pred_std   = pred_flat.std(unbiased=False).clamp(min=1e-8)
    target_std = target_flat.std(unbiased=False).detach().clamp(min=return_scale * 0.1)
    spread_ratio      = pred_std / target_std
    dispersion_penalty = (
        F.relu(min_dispersion_ratio - spread_ratio)
        + F.relu(spread_ratio - max_dispersion_ratio)
    )

    with torch.no_grad():
        dir_acc  = ((pred_flat > 0) == (target_flat > 0)).float().mean()
        exact_ic = exact_spearman_ic(pred_flat, target_flat)

    loss = (
        mse_weight        * mse
        + ic_weight       * ic_loss
        + dispersion_weight * dispersion_penalty
    )
    metrics = {
        "mse":        float(mse.detach().cpu()),
        "ic":         float(soft_corr.detach().cpu()),
        "rank_ic":    float(exact_ic.detach().cpu()),
        "pred_std":   float(pred_std.detach().cpu()),
        "target_std": float(target_std.detach().cpu()),
        "disp_pen":   float(dispersion_penalty.detach().cpu()),
        "dir_acc":    float(dir_acc.cpu()),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    *,
    ic_weight_override: float | None = None,
    temperature: float = 0.05,
    return_scale: float = 0.02,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(mode=training)

    sums  = {k: 0.0 for k in ("loss", "mse", "ic", "rank_ic",
                               "pred_std", "target_std", "disp_pen", "dir_acc")}
    steps = 0

    for batch in loader:
        sample_list = batch if isinstance(batch, list) else [batch]
        for data in sample_list:
            pos_adj, neg_adj, features, labels, mask = extract_data(data, str(device))

            if labels.dim() > 1 and labels.shape[-1] > 1:
                labels = labels[:, args.target_horizon]

            mask_t = _mask_to_tensor(mask, len(labels), device)
            logits = model(features, pos_adj, neg_adj)

            logits = _ensure_2d(logits)
            labels = _ensure_2d(labels)
            pred   = logits[mask_t]
            target = labels[mask_t]
            if pred.numel() == 0 or target.numel() == 0:
                continue

            pred = pred.clamp(-10.0 * return_scale, 10.0 * return_scale)

            eff_ic = args.ic_weight if ic_weight_override is None else ic_weight_override
            loss, metrics = composite_loss(
                pred, target,
                mse_weight=args.mse_weight,
                ic_weight=eff_ic,
                dispersion_weight=args.dispersion_weight,
                min_dispersion_ratio=args.min_dispersion_ratio,
                max_dispersion_ratio=args.max_dispersion_ratio,
                temperature=temperature,
                return_scale=return_scale,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            sums["loss"] += float(loss.detach().cpu())
            for k in ("mse", "ic", "rank_ic", "pred_std", "target_std", "disp_pen", "dir_acc"):
                sums[k] += metrics[k]
            steps += 1

    if steps == 0:
        return {k: 0.0 for k in sums}
    return {k: v / steps for k, v in sums.items()}


def _make_lr_lambda(warmup_epochs: int, total_epochs: int):
    def _fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return _fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted([p.name for p in args.data_dir.glob("*.pkl")])
    if not files:
        raise RuntimeError(f"No .pkl files found in {args.data_dir}")
    split = compute_split_indices(files, args)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"Train: {files[split.train_start]} → {files[split.train_end_exclusive - 1]} "
        f"({split.train_end_exclusive - split.train_start} samples)"
    )
    print(
        f"Test:  {files[split.test_start]} → {files[split.test_end_exclusive - 1]} "
        f"({split.test_end_exclusive - split.test_start} samples)"
    )

    train_ds = AllGraphDataSampler(
        base_dir=str(args.data_dir),
        data_start=split.train_start,
        data_middle=split.train_end_exclusive,
        data_end=split.test_end_exclusive,
    )
    test_ds = AllGraphDataSampler(
        base_dir=str(args.data_dir),
        mode="val",
        data_start=split.train_start,
        data_middle=split.test_start,
        data_end=split.test_end_exclusive,
    )
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("Train/test dataset empty after split filtering.")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=1, pin_memory=pin,
                              collate_fn=lambda x: x)
    test_loader  = DataLoader(test_ds,  batch_size=1, pin_memory=pin,
                              collate_fn=lambda x: x)

    model = HybridStockModel(
        in_features=args.in_features,
        embed_dim=args.embed_dim,
        num_mage_layers=args.num_mage_layers,
        num_moe_experts=args.num_moe_experts,
        num_mha_heads=args.num_mha_heads,
        gat_heads=args.gat_heads,
        gat_out_features=args.gat_out_features,
        num_hyper_edges=args.num_hyper_edges,
        dropout=args.dropout,
        predictor_out_dim=1,
        predictor_activation=None,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Hybrid model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_make_lr_lambda(2, args.epochs)
    )

    best_test_ic  = -np.inf
    best_test_mse = np.inf
    best_epoch    = -1
    wait          = 0
    best_path     = args.model_dir / f"{split.pre_data}_hybrid_best.dat"
    history       = {"epoch": [], "train_loss": [], "test_loss": []}

    pbar = tqdm(range(1, args.epochs + 1), desc="Hybrid IC-ranked epochs", unit="epoch")
    for epoch in pbar:
        ic_ramp          = min(1.0, epoch / max(1, args.ic_warmup_epochs))
        current_ic_weight = args.ic_weight * ic_ramp
        temperature      = max(0.02, 0.2 * (0.95 ** epoch))

        train_m = run_epoch(train_loader, model, device, optimizer, args,
                            ic_weight_override=current_ic_weight,
                            temperature=temperature,
                            return_scale=args.return_scale)
        test_m  = run_epoch(test_loader,  model, device, None, args,
                            ic_weight_override=current_ic_weight,
                            temperature=temperature,
                            return_scale=args.return_scale)
        scheduler.step()

        pbar.set_postfix(
            tr_mse=f"{train_m['mse']:.4f}",
            te_mse=f"{test_m['mse']:.4f}",
            te_ic =f"{test_m['rank_ic']:.4f}",
            te_dir=f"{test_m['dir_acc']:.3f}",
            spread=f"{test_m['pred_std'] / max(test_m['target_std'], 1e-8):.3f}",
        )
        history["epoch"].append(epoch)
        history["train_loss"].append(train_m["loss"])
        history["test_loss"].append(test_m["loss"])

        improved = test_m["rank_ic"] > best_test_ic or (
            np.isclose(test_m["rank_ic"], best_test_ic)
            and test_m["mse"] < best_test_mse
        )
        if improved:
            best_test_ic  = test_m["rank_ic"]
            best_test_mse = test_m["mse"]
            best_epoch    = epoch
            wait          = 0
            torch.save(
                {
                    "model":       model.state_dict(),
                    "epoch":       epoch,
                    "test_ic":     best_test_ic,
                    "test_soft_ic": test_m["ic"],
                    "test_mse":    best_test_mse,
                    "config":      vars(args),
                    "split":       split.__dict__,
                },
                best_path,
            )
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
                break

    # Reload best checkpoint for final evaluation
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    final_train = run_epoch(train_loader, model, device, None, args,
                            return_scale=args.return_scale)
    final_test  = run_epoch(test_loader,  model, device, None, args,
                            return_scale=args.return_scale)

    # Loss curve plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["epoch"], history["test_loss"],  label="Test Loss",  linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Composite Loss")
    plt.title("Hybrid THGNN×MaGNet — IC-Ranked Training/Test Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = PLOT_DIR / f"{split.pre_data}_hybrid_loss_curve.png"
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)

    print("\nTraining complete.")
    print(f"Best epoch:    {best_epoch}")
    print(f"Best test IC:  {best_test_ic:.4f}")
    print(f"Best test MSE: {best_test_mse:.6f}")
    print(f"Checkpoint:    {best_path}")
    print(
        f"Final metrics | "
        f"train_loss={final_train['loss']:.6f}  "
        f"test_loss={final_test['loss']:.6f}  "
        f"test_rank_ic={final_test['rank_ic']:.4f}  "
        f"dir_acc={final_test['dir_acc']:.4f}"
    )
    print(f"Loss plot:     {plot_path}")


if __name__ == "__main__":
    main()
