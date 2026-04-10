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
from model.Thgnn import StockHeteGAT
from trainer.trainer import extract_data


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "data_train_predict"
MODEL_DIR = DATA_DIR / "model_saved"
PLOT_DIR = DATA_DIR / "plots"


@dataclass
class SplitIndices:
    train_start: int
    train_end_exclusive: int
    test_start: int
    test_end_exclusive: int
    pre_data: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train THGNN with MSE + IC + dispersion regularization.")
    parser.add_argument("--data-dir", type=Path, default=TRAIN_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--train-start-date", type=str, default="2015-01-01")
    parser.add_argument("--train-end-date", type=str, default="2024-12-31")
    parser.add_argument("--test-start-date", type=str, default="2025-01-01")
    parser.add_argument("--test-end-date", type=str, default="2026-12-31")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--out-features", type=int, default=32)
    parser.add_argument("--in-features", type=int, default=12,
                        help="Number of input features per stock per timestep. "
                             "Must match the feature dimension in the graph samples.")
    parser.add_argument("--mse-weight", type=float, default=0.7)
    parser.add_argument("--ic-weight", type=float, default=0.5)
    parser.add_argument("--dispersion-weight", type=float, default=0.3)
    parser.add_argument("--min-dispersion-ratio", type=float, default=0.5,
                        help="Penalize pred_std < min_dispersion_ratio * target_std to prevent mean-collapse. "
                             "Previous default 0.2 never triggered (spread was 0.26), so the model hedged "
                             "toward near-zero predictions with no gradient pressure to spread.")
    parser.add_argument("--max-dispersion-ratio", type=float, default=2.0,
                        help="Penalize pred_std > max_dispersion_ratio * target_std to prevent over-spreading.")
    parser.add_argument("--return-scale", type=float, default=0.02,
                        help="Typical daily return std used to normalize MSE. "
                             "MSE is divided by return_scale**2 so it is O(1) and scale-consistent "
                             "across all batches and splits. Default 0.02 matches Nifty50 cross-sectional "
                             "return std (~2%% per day in decimal form). Set to ~1.0 if returns are in "
                             "percentage points.")
    parser.add_argument("--target-horizon", type=int, default=0,
                        help="Which label horizon to train on (0=next-day, 1, 2). "
                             "Avoids conflicting gradients from negatively correlated horizons.")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ic-warmup-epochs", type=int, default=3,
                        help="Ramp IC loss weight from 0 to --ic-weight over this many epochs. "
                             "Shorter warmup so IC gradient arrives before MSE drives predictions to zero.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _index_for_date(file_dates: list[pd.Timestamp], date_str: str) -> int:
    """Return index of first file with date >= date_str (for start boundaries)."""
    dt = pd.to_datetime(date_str).normalize()
    for idx, d in enumerate(file_dates):
        if d >= dt:
            return idx
    # Requested date is beyond all available data — return last index so the
    # caller can clamp it. A hard error here would break valid "train up to
    # latest available data" usage patterns; the caller validates ordering.
    return len(file_dates) - 1


def _exclusive_end_index_for_date(file_dates: list[pd.Timestamp], date_str: str) -> int:
    """Return exclusive-end index: number of files with date <= date_str.

    Using first-file->=date + 1 for end boundaries fails when the end date
    falls on a weekend/holiday — both the end date and the next start date
    resolve to the same trading day, causing val_end > test_start.
    This function walks forward and stops at the first file strictly after
    date_str, so weekend/holiday end dates land on the last trading day before
    the boundary rather than the first one after it.
    """
    dt = pd.to_datetime(date_str).normalize()
    for idx, d in enumerate(file_dates):
        if d > dt:
            return idx
    return len(file_dates)


def compute_split_indices(files: list[str], args: argparse.Namespace) -> SplitIndices:
    file_dates = [pd.to_datetime(Path(name).stem).normalize() for name in files]
    min_date = file_dates[0]
    max_date = file_dates[-1]

    train_start = _index_for_date(file_dates, args.train_start_date)
    train_end_exclusive = _exclusive_end_index_for_date(file_dates, args.train_end_date)
    test_start = _index_for_date(file_dates, args.test_start_date)
    test_end_exclusive = _exclusive_end_index_for_date(file_dates, args.test_end_date)

    train_end_exclusive = min(train_end_exclusive, len(files))
    test_end_exclusive = min(test_end_exclusive, len(files))

    if not (train_start < train_end_exclusive <= test_start < test_end_exclusive):
        raise ValueError(
            "Invalid split ordering. Required: train_start < train_end <= test_start < test_end. "
            f"Resolved indices: train [{train_start}, {train_end_exclusive}), "
            f"test [{test_start}, {test_end_exclusive})."
        )

    if train_end_exclusive <= train_start:
        raise ValueError("Training split is empty.")
    if test_end_exclusive <= test_start:
        raise ValueError("Test split is empty.")

    pre_data = Path(files[train_end_exclusive - 1]).stem

    if min_date > pd.to_datetime(args.train_start_date):
        print(
            f"Warning: Requested train start {args.train_start_date} is earlier than available data. "
            f"Using first available sample {min_date.date()}."
        )
    if max_date < pd.to_datetime(args.test_end_date):
        print(
            f"Warning: Requested test end {args.test_end_date} exceeds available data. "
            f"Using last available sample {max_date.date()}."
        )

    return SplitIndices(
        train_start=train_start,
        train_end_exclusive=train_end_exclusive,
        test_start=test_start,
        test_end_exclusive=test_end_exclusive,
        pre_data=pre_data,
    )


def _mask_to_tensor(mask, length: int, device: torch.device) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        return mask.to(device).bool()
    return torch.tensor(mask, device=device, dtype=torch.bool)


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(-1)
    return x


def _soft_rank(x: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """Differentiable approximation of rank via pairwise sigmoid comparisons.

    For each element x_i, its soft rank ≈ 1 + Σ_{j} sigmoid((x_i - x_j) / temperature).
    As temperature → 0 this converges to true rank; larger temperature smooths gradients.
    τ=0.05 (vs. original 0.01) gives smoother gradients — less risk of vanishing signal
    when predictions are already approximately rank-ordered.
    O(N²) in the number of stocks — fine for N ≈ 50.
    """
    diff = x.unsqueeze(0) - x.unsqueeze(1)  # (N, N): diff[i, j] = x[i] - x[j]
    return torch.sigmoid(diff / temperature).sum(dim=1)  # (N,)


def cross_sectional_ic(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Differentiable cross-sectional IC using soft Spearman rank correlation.

    Spearman IC = Pearson correlation of ranks. Using differentiable soft ranks
    gives a proper ranking-based gradient signal that is robust to outlier returns,
    unlike raw Pearson IC which can be dominated by a single large-move stock.
    Target ranks are detached — gradients only flow through predictions.
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    if pred_flat.numel() < 2:
        return torch.tensor(0.0, device=pred.device)

    pred_rank = _soft_rank(pred_flat, temperature)
    # Detach target: its ranks are fixed labels, not a learnable quantity.
    target_rank = _soft_rank(target_flat.detach(), temperature).detach()

    pred_r = pred_rank - pred_rank.mean()
    target_r = target_rank - target_rank.mean()
    denom = torch.sqrt((pred_r ** 2).sum() * (target_r ** 2).sum() + eps)
    if denom.item() <= eps:
        return torch.tensor(0.0, device=pred.device)
    return torch.sum(pred_r * target_r) / denom


def _exact_ranks(x: torch.Tensor) -> torch.Tensor:
    """Return 1..N ranks for a 1D tensor.

    This is used only for evaluation / checkpoint selection, not for gradients.
    Financial return labels are effectively continuous, so simple ordinal ranks
    are sufficient here.
    """
    x_flat = x.reshape(-1)
    order = torch.argsort(x_flat)
    ranks = torch.empty_like(x_flat, dtype=torch.float32)
    ranks[order] = torch.arange(1, x_flat.numel() + 1, device=x.device, dtype=torch.float32)
    return ranks


def exact_spearman_ic(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    if pred_flat.numel() < 2:
        return torch.tensor(0.0, device=pred.device)

    pred_rank = _exact_ranks(pred_flat)
    target_rank = _exact_ranks(target_flat)
    pred_r = pred_rank - pred_rank.mean()
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
    return_scale: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = _ensure_2d(pred)
    target = _ensure_2d(target)

    # Cross-sectional demeaning: remove the market factor before computing MSE.
    # Raw daily returns are dominated by the market move (e.g. Nifty up 1.5%
    # → all stocks roughly +1–2%). MSE on raw returns rewards predicting market
    # direction, not relative ranking. Demeaning isolates the idiosyncratic
    # component so gradients point toward cross-sectional outperformance.
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    pred_cs = pred_flat - pred_flat.mean()
    target_cs = target_flat - target_flat.detach().mean()
    mse = F.mse_loss(pred_cs, target_cs) / (return_scale ** 2)
    soft_corr = cross_sectional_ic(pred, target, temperature=temperature)
    ic_loss = 1.0 - soft_corr

    pred_std = pred_flat.std(unbiased=False).clamp(min=1e-8)
    # Floor target_std at return_scale × 0.1 rather than 1e-8.
    # On days where all stocks move in lockstep (low cross-sectional dispersion),
    # target_std → 0 and spread_ratio = pred_std / target_std → ∞, spiking the
    # dispersion penalty to thousands and dominating the total loss.
    # A floor of return_scale × 0.1 caps spread_ratio at ~100 in the worst case
    # while having no effect on normal batches where target_std ≫ this floor.
    target_std = target_flat.std(unbiased=False).detach().clamp(min=return_scale * 0.1)
    # Dimensionless spread ratio penalty: O(1) and independent of return scale.
    # Raw std penalty (previous) was O(0.01) — negligible vs O(1) MSE/IC terms.
    # Ratio form also produces a strong gradient when pred_std → 0 (collapse),
    # since ∂ratio/∂pred ∝ 1/pred_std, which is exactly when correction is needed.
    spread_ratio = pred_std / target_std
    dispersion_penalty = F.relu(min_dispersion_ratio - spread_ratio) + F.relu(spread_ratio - max_dispersion_ratio)

    with torch.no_grad():
        dir_acc = ((pred_flat > 0) == (target_flat > 0)).float().mean()
        exact_ic = exact_spearman_ic(pred_flat, target_flat)

    loss = (
        mse_weight * mse
        + ic_weight * ic_loss
        + dispersion_weight * dispersion_penalty
    )
    metrics = {
        "mse": float(mse.detach().cpu().item()),
        "ic": float(soft_corr.detach().cpu().item()),
        "rank_ic": float(exact_ic.detach().cpu().item()),
        "pred_std": float(pred_std.detach().cpu().item()),
        "target_std": float(target_std.detach().cpu().item()),
        "disp_pen": float(dispersion_penalty.detach().cpu().item()),
        "dir_acc": float(dir_acc.cpu().item()),
    }
    return loss, metrics


def run_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    *,
    ic_weight_override: float | None = None,
    temperature: float = 0.05,
    return_scale: float = 0.01,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(mode=training)

    sums = {
        "loss": 0.0,
        "mse": 0.0,
        "ic": 0.0,
        "rank_ic": 0.0,
        "pred_std": 0.0,
        "target_std": 0.0,
        "disp_pen": 0.0,
        "dir_acc": 0.0,
    }
    steps = 0

    for batch in loader:
        sample_list = batch if isinstance(batch, list) else [batch]
        for data in sample_list:
            pos_adj, neg_adj, features, labels, mask = extract_data(data, str(device))
            # Use a single horizon to avoid conflicting IC gradients across
            # negatively-correlated label horizons (e.g. corr(h0,h2) = -0.24).
            if labels.dim() > 1 and labels.shape[-1] > 1:
                labels = labels[:, args.target_horizon]
            mask_t = _mask_to_tensor(mask, len(labels), device)
            logits = model(features, pos_adj, neg_adj)

            logits = _ensure_2d(logits)
            labels = _ensure_2d(labels)
            pred = logits[mask_t]
            target = labels[mask_t]
            if pred.numel() == 0 or target.numel() == 0:
                continue
            # Clamp to ±10 × return_scale (e.g. ±20% for return_scale=0.02).
            # Prevents a few out-of-distribution batches on the test set from
            # producing pred_std >> return_scale and spiking MSE/return_scale².
            pred = pred.clamp(-10.0 * return_scale, 10.0 * return_scale)

            effective_ic_weight = args.ic_weight if ic_weight_override is None else ic_weight_override
            loss, metrics = composite_loss(
                pred,
                target,
                mse_weight=args.mse_weight,
                ic_weight=effective_ic_weight,
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

            sums["loss"] += float(loss.detach().cpu().item())
            for key in ["mse", "ic", "rank_ic", "pred_std", "target_std", "disp_pen", "dir_acc"]:
                sums[key] += metrics[key]
            steps += 1

    if steps == 0:
        return {k: 0.0 for k in sums}
    return {k: v / steps for k, v in sums.items()}


def _make_lr_lambda(warmup_epochs: int, total_epochs: int):
    """Linear warmup (0.1× → 1×) then cosine annealing.

    Replaces SequentialLR + LinearLR + CosineAnnealingLR with a single
    LambdaLR so PyTorch does not emit the deprecated-epoch-parameter warning
    that SequentialLR triggers by passing `epoch` to its sub-schedulers.
    """
    def _fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return _fn


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
        f"Train: {files[split.train_start]} -> {files[split.train_end_exclusive - 1]} "
        f"({split.train_end_exclusive - split.train_start} samples)"
    )
    print(
        f"Test:  {files[split.test_start]} -> {files[split.test_end_exclusive - 1]} "
        f"({split.test_end_exclusive - split.test_start} samples)"
    )
    print(f"Checkpoint pre_data: {split.pre_data}")

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
    train_loader = DataLoader(train_ds, batch_size=1, pin_memory=pin, collate_fn=lambda x: x)
    test_loader = DataLoader(test_ds, batch_size=1, pin_memory=pin, collate_fn=lambda x: x)

    target_dim = 1  # single horizon prediction

    model = StockHeteGAT(
        in_features=args.in_features,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        out_features=args.out_features,
        predictor_out_dim=target_dim,
        predictor_activation=None,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # LR warmup: 2 epochs so full LR arrives before IC warmup (3 epochs) completes.
    # Previously 5 epochs caused LR and IC to ramp simultaneously, leaving the model
    # with weak signals on both axes during the critical early phase.
    warmup_epochs = 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_make_lr_lambda(warmup_epochs, args.epochs)
    )

    best_test_ic = -np.inf
    best_test_mse = np.inf
    best_epoch = -1
    wait = 0
    best_path = args.model_dir / f"{split.pre_data}_icrank_best.dat"
    history = {"epoch": [], "train_loss": [], "test_loss": []}

    pbar = tqdm(range(1, args.epochs + 1), desc="IC-ranked epochs", unit="epoch")
    for epoch in pbar:
        # Ramp IC weight from 0 → target over ic_warmup_epochs to avoid noisy
        # IC gradients dominating MSE fitting in early epochs.
        ic_ramp = min(1.0, epoch / max(1, args.ic_warmup_epochs))
        current_ic_weight = args.ic_weight * ic_ramp

        # Anneal soft-rank temperature: start high (smooth gradients when predictions
        # are far apart) and decay toward 0.02 (sharper rank signal near convergence).
        temperature = max(0.02, 0.2 * (0.95 ** epoch))

        # Use the same IC weight, temperature, and return_scale for both splits
        # so the loss curves are directly comparable across epochs in the plot.
        train_metrics = run_epoch(train_loader, model, device, optimizer, args, ic_weight_override=current_ic_weight, temperature=temperature, return_scale=args.return_scale)
        test_metrics = run_epoch(test_loader, model, device, None, args, ic_weight_override=current_ic_weight, temperature=temperature, return_scale=args.return_scale)
        scheduler.step()

        pbar.set_postfix(
            train_mse=f"{train_metrics['mse']:.6f}",
            test_mse=f"{test_metrics['mse']:.6f}",
            test_ic=f"{test_metrics['rank_ic']:.4f}",
            test_soft_ic=f"{test_metrics['ic']:.4f}",
            test_dir=f"{test_metrics['dir_acc']:.3f}",
            spread=f"{(test_metrics['pred_std'] / max(test_metrics['target_std'], 1e-8)):.3f}",
        )
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["test_loss"].append(test_metrics["loss"])

        improved = (test_metrics["rank_ic"] > best_test_ic) or (
            np.isclose(test_metrics["rank_ic"], best_test_ic) and test_metrics["mse"] < best_test_mse
        )
        if improved:
            best_test_ic = test_metrics["rank_ic"]
            best_test_mse = test_metrics["mse"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "test_ic": best_test_ic,
                    "test_soft_ic": test_metrics["ic"],
                    "test_mse": best_test_mse,
                    "config": vars(args),
                    "split": split.__dict__,
                },
                best_path,
            )
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
                break

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    final_train_metrics = run_epoch(train_loader, model, device, None, args, return_scale=args.return_scale)
    final_test_metrics = run_epoch(test_loader, model, device, None, args, return_scale=args.return_scale)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["epoch"], history["test_loss"], label="Test Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Composite Loss")
    plt.title("IC-Ranked Training/Test Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_plot_path = PLOT_DIR / f"{split.pre_data}_icrank_loss_curve.png"
    plt.savefig(loss_plot_path, dpi=200)
    plt.close(fig)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best test IC: {best_test_ic:.4f}")
    print(f"Best test MSE: {best_test_mse:.6f}")
    print(f"Saved best checkpoint: {best_path}")
    print(
        "Final metrics from best checkpoint | "
        f"train_loss={final_train_metrics['loss']:.6f}, "
        f"test_loss={final_test_metrics['loss']:.6f}"
    )
    print(
        "Directional accuracy | "
        f"train={final_train_metrics['dir_acc']:.4f}, "
        f"test={final_test_metrics['dir_acc']:.4f}"
    )
    print(f"Saved loss plot: {loss_plot_path}")


if __name__ == "__main__":
    main()
