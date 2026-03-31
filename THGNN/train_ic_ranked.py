from __future__ import annotations

import argparse
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
    val_start: int
    val_end_exclusive: int
    test_start: int
    test_end_exclusive: int
    pre_data: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train THGNN with MSE + IC + dispersion regularization.")
    parser.add_argument("--data-dir", type=Path, default=TRAIN_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--train-start-date", type=str, default="2015-01-01")
    parser.add_argument("--train-end-date", type=str, default="2023-12-29")
    parser.add_argument("--val-start-date", type=str, default="2024-01-01")
    parser.add_argument("--val-end-date", type=str, default="2024-12-31")
    parser.add_argument("--test-start-date", type=str, default="2025-01-01")
    parser.add_argument("--test-end-date", type=str, default="2026-2-28")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--out-features", type=int, default=32)
    parser.add_argument("--in-features", type=int, default=12,
                        help="Number of input features per stock per timestep. "
                             "Must match the feature dimension in the graph samples.")
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--ic-weight", type=float, default=0.35)
    parser.add_argument("--dispersion-weight", type=float, default=0.2)
    parser.add_argument("--min-dispersion-ratio", type=float, default=0.2)
    parser.add_argument("--max-dispersion-ratio", type=float, default=2.0,
                        help="Penalize pred_std > max_dispersion_ratio * target_std to prevent over-spreading.")
    parser.add_argument("--target-horizon", type=int, default=0,
                        help="Which label horizon to train on (0=next-day, 1, 2). "
                             "Avoids conflicting gradients from negatively correlated horizons.")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--ic-warmup-epochs", type=int, default=10,
                        help="Ramp IC loss weight from 0 to --ic-weight over this many epochs. "
                             "Prevents noisy IC gradients from dominating before MSE fitting.")
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
    val_start = _index_for_date(file_dates, args.val_start_date)
    val_end_exclusive = _exclusive_end_index_for_date(file_dates, args.val_end_date)
    test_start = _index_for_date(file_dates, args.test_start_date)
    test_end_exclusive = _exclusive_end_index_for_date(file_dates, args.test_end_date)

    val_end_exclusive = min(val_end_exclusive, len(files))
    test_end_exclusive = min(test_end_exclusive, len(files))

    if train_end_exclusive > len(files):
        train_end_exclusive = len(files)

    if not (train_start < train_end_exclusive <= val_start < val_end_exclusive <= test_start < test_end_exclusive):
        raise ValueError(
            "Invalid split ordering. Required: "
            "train_start < train_end <= val_start < val_end <= test_start < test_end. "
            f"Resolved indices: train [{train_start}, {train_end_exclusive}), "
            f"val [{val_start}, {val_end_exclusive}), "
            f"test [{test_start}, {test_end_exclusive})."
        )

    if train_end_exclusive <= train_start:
        raise ValueError("Training split is empty.")
    if val_end_exclusive <= val_start:
        raise ValueError("Validation split is empty.")
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
        val_start=val_start,
        val_end_exclusive=val_end_exclusive,
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
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = _ensure_2d(pred)
    target = _ensure_2d(target)

    mse = F.mse_loss(pred, target)
    soft_corr = cross_sectional_ic(pred, target)
    ic_loss = 1.0 - soft_corr

    pred_std = pred.reshape(-1).std(unbiased=False)
    target_std = target.reshape(-1).std(unbiased=False).detach()
    min_std = min_dispersion_ratio * target_std
    max_std = max_dispersion_ratio * target_std
    # Penalize both under- and over-spreading relative to target return distribution.
    dispersion_penalty = F.relu(min_std - pred_std) + F.relu(pred_std - max_std)

    with torch.no_grad():
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
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

            effective_ic_weight = args.ic_weight if ic_weight_override is None else ic_weight_override
            loss, metrics = composite_loss(
                pred,
                target,
                mse_weight=args.mse_weight,
                ic_weight=effective_ic_weight,
                dispersion_weight=args.dispersion_weight,
                min_dispersion_ratio=args.min_dispersion_ratio,
                max_dispersion_ratio=args.max_dispersion_ratio,
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted([p.name for p in args.data_dir.glob("*.pkl")])
    if not files:
        raise RuntimeError(f"No .pkl files found in {args.data_dir}")
    split = compute_split_indices(files, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"Device: {device}")
    print(
        f"Train: {files[split.train_start]} -> {files[split.train_end_exclusive - 1]} "
        f"({split.train_end_exclusive - split.train_start} samples)"
    )
    print(
        f"Val:   {files[split.val_start]} -> {files[split.val_end_exclusive - 1]} "
        f"({split.val_end_exclusive - split.val_start} samples)"
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
        data_end=split.val_end_exclusive,
    )
    val_ds = AllGraphDataSampler(
        base_dir=str(args.data_dir),
        mode="val",
        data_start=split.train_start,
        data_middle=split.val_start,
        data_end=split.val_end_exclusive,
    )
    test_ds = AllGraphDataSampler(
        base_dir=str(args.data_dir),
        mode="val",
        data_start=split.train_start,
        data_middle=split.test_start,
        data_end=split.test_end_exclusive,
    )
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("Train/val/test dataset empty after split filtering.")

    train_loader = DataLoader(train_ds, batch_size=1, pin_memory=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_ds, batch_size=1, pin_memory=True, collate_fn=lambda x: x)
    test_loader = DataLoader(test_ds, batch_size=1, pin_memory=True, collate_fn=lambda x: x)

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
    warmup_epochs = 5
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    best_val_ic = -np.inf
    best_val_mse = np.inf
    best_epoch = -1
    wait = 0
    best_path = args.model_dir / f"{split.pre_data}_icrank_best.dat"
    history = {"epoch": [], "train_loss": [], "val_loss": [], "test_loss": []}

    pbar = tqdm(range(1, args.epochs + 1), desc="IC-ranked epochs", unit="epoch")
    for epoch in pbar:
        # Ramp IC weight from 0 → target over ic_warmup_epochs to avoid noisy
        # IC gradients dominating MSE fitting in early epochs.
        ic_ramp = min(1.0, epoch / max(1, args.ic_warmup_epochs))
        current_ic_weight = args.ic_weight * ic_ramp

        train_metrics = run_epoch(train_loader, model, device, optimizer, args, ic_weight_override=current_ic_weight)
        val_metrics = run_epoch(val_loader, model, device, None, args)
        test_metrics = run_epoch(test_loader, model, device, None, args)
        scheduler.step()

        pbar.set_postfix(
            train_mse=f"{train_metrics['mse']:.6f}",
            val_mse=f"{val_metrics['mse']:.6f}",
            val_ic=f"{val_metrics['rank_ic']:.4f}",
            val_soft_ic=f"{val_metrics['ic']:.4f}",
            val_dir=f"{val_metrics['dir_acc']:.3f}",
            test_dir=f"{test_metrics['dir_acc']:.3f}",
            spread=f"{(val_metrics['pred_std'] / max(val_metrics['target_std'], 1e-8)):.3f}",
        )
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["test_loss"].append(test_metrics["loss"])

        improved = (val_metrics["rank_ic"] > best_val_ic) or (
            np.isclose(val_metrics["rank_ic"], best_val_ic) and val_metrics["mse"] < best_val_mse
        )
        if improved:
            best_val_ic = val_metrics["rank_ic"]
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_ic": best_val_ic,
                    "val_soft_ic": val_metrics["ic"],
                    "val_mse": best_val_mse,
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
    final_train_metrics = run_epoch(train_loader, model, device, None, args)
    final_val_metrics = run_epoch(val_loader, model, device, None, args)
    final_test_metrics = run_epoch(test_loader, model, device, None, args)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss", linewidth=2)
    plt.plot(history["epoch"], history["test_loss"], label="Test Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Composite Loss")
    plt.title("IC-Ranked Training/Validation/Test Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_plot_path = PLOT_DIR / f"{split.pre_data}_icrank_loss_curve.png"
    plt.savefig(loss_plot_path, dpi=200)
    plt.close(fig)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val IC: {best_val_ic:.4f}")
    print(f"Best val MSE: {best_val_mse:.6f}")
    print(f"Saved best checkpoint: {best_path}")
    print(
        "Final metrics from best checkpoint | "
        f"train_loss={final_train_metrics['loss']:.6f}, "
        f"val_loss={final_val_metrics['loss']:.6f}, "
        f"test_loss={final_test_metrics['loss']:.6f}"
    )
    print(
        "Directional accuracy | "
        f"train={final_train_metrics['dir_acc']:.4f}, "
        f"val={final_val_metrics['dir_acc']:.4f}, "
        f"test={final_test_metrics['dir_acc']:.4f}"
    )
    print(f"Saved loss plot: {loss_plot_path}")


if __name__ == "__main__":
    main()
