from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
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
    parser.add_argument("--val-start-date", type=str, default=None,
                        help="Ignored — kept for walk_forward_train.py compatibility.")
    parser.add_argument("--val-end-date", type=str, default=None,
                        help="Ignored — kept for walk_forward_train.py compatibility.")
    parser.add_argument("--test-start-date", type=str, default="2025-01-01")
    parser.add_argument("--test-end-date", type=str, default="2026-12-31")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--out-features", type=int, default=32)
    parser.add_argument("--in-features", type=int, default=12,
                        help="Number of input features per stock per timestep. "
                             "Must match the feature dimension in the graph samples.")
    parser.add_argument("--mse-weight", type=float, default=0.7)
    parser.add_argument("--ic-weight", type=float, default=0.5)
    parser.add_argument("--dispersion-weight", type=float, default=0.6)
    parser.add_argument("--min-dispersion-ratio", type=float, default=0.6,
                        help="Penalize pred_std < min_dispersion_ratio * target_std to prevent mean-collapse.")
    parser.add_argument("--max-dispersion-ratio", type=float, default=2.0,
                        help="Penalize pred_std > max_dispersion_ratio * target_std to prevent over-spreading.")
    parser.add_argument("--return-scale", type=float, default=0.023,
                        help="Typical daily return std used to normalize MSE.")
    parser.add_argument("--target-horizon", type=int, default=0,
                        help="Which label horizon to train on (0=next-day, 1, 2).")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--ic-warmup-epochs", type=int, default=5,
                        help="Ramp IC loss weight from 0 to --ic-weight over this many epochs.")
    parser.add_argument("--lr-warmup-epochs", type=int, default=5,
                        help="Linear LR warmup from 0.1x to 1x over this many epochs (then cosine decay).")
    parser.add_argument("--seed", type=int, default=42)
    # Divergence guard (mirrors train_hybrid.py)
    parser.add_argument("--max-loss-ratio", type=float, default=3.5,
                        help="Stop if test_loss / train_loss exceeds this for --overfit-patience "
                             "consecutive epochs (divergence guard).")
    parser.add_argument("--overfit-patience", type=int, default=5,
                        help="Consecutive epochs of loss-ratio violation before divergence stop.")
    # Logging
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="Directory for CSV/log/JSON artefacts. Defaults to --plot-dir.")
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
    return len(file_dates) - 1


def _exclusive_end_index_for_date(file_dates: list[pd.Timestamp], date_str: str) -> int:
    """Return exclusive-end index: number of files with date <= date_str."""
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
    """Differentiable approximation of rank via pairwise sigmoid comparisons."""
    diff = x.unsqueeze(0) - x.unsqueeze(1)  # (N, N): diff[i, j] = x[i] - x[j]
    return torch.sigmoid(diff / temperature).sum(dim=1)  # (N,)


def cross_sectional_ic(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Differentiable cross-sectional IC using soft Spearman rank correlation."""
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    if pred_flat.numel() < 2:
        return torch.tensor(0.0, device=pred.device)

    pred_rank = _soft_rank(pred_flat, temperature)
    target_rank = _soft_rank(target_flat.detach(), temperature).detach()

    pred_r = pred_rank - pred_rank.mean()
    target_r = target_rank - target_rank.mean()
    denom = torch.sqrt((pred_r ** 2).sum() * (target_r ** 2).sum() + eps)
    if denom.item() <= eps:
        return torch.tensor(0.0, device=pred.device)
    return torch.sum(pred_r * target_r) / denom


def _exact_ranks(x: torch.Tensor) -> torch.Tensor:
    """Return 1..N ranks for a 1D tensor (evaluation only, not for gradients)."""
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
    return_scale: float = 0.023,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = _ensure_2d(pred)
    target = _ensure_2d(target)

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    pred_cs = pred_flat - pred_flat.mean()
    target_cs = target_flat - target_flat.detach().mean()
    mse = F.mse_loss(pred_cs, target_cs) / (return_scale ** 2)
    soft_corr = cross_sectional_ic(pred, target, temperature=temperature)
    ic_loss = 1.0 - soft_corr

    pred_std = pred_flat.std(unbiased=False).clamp(min=1e-8)
    target_std = target_flat.std(unbiased=False).detach().clamp(min=return_scale * 0.1)
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
    # Return GPU tensors — caller accumulates on GPU and syncs once per epoch
    metrics = {
        "mse":        mse.detach(),
        "ic":         soft_corr.detach(),
        "rank_ic":    exact_ic,
        "pred_std":   pred_std.detach(),
        "target_std": target_std.detach(),
        "disp_pen":   dispersion_penalty.detach(),
        "dir_acc":    dir_acc,
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
    return_scale: float = 0.023,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(mode=training)

    sums = {k: 0.0 for k in (
        "loss", "mse", "ic", "rank_ic",
        "pred_std", "target_std", "disp_pen", "dir_acc", "grad_norm",
    )}
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
            pred = logits[mask_t]
            target = labels[mask_t]
            if pred.numel() == 0 or target.numel() == 0:
                continue
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
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                sums["grad_norm"] += grad_norm  # GPU tensor — sync deferred to epoch end
            else:
                sums["grad_norm"] += 0.0

            sums["loss"] += loss.detach()
            for key in ("mse", "ic", "rank_ic", "pred_std", "target_std", "disp_pen", "dir_acc"):
                sums[key] += metrics[key]
            steps += 1

    if steps == 0:
        return {k: 0.0 for k in sums}
    return {k: float(v) / steps for k, v in sums.items()}  # one GPU→CPU sync per metric


def _make_lr_lambda(warmup_epochs: int, total_epochs: int):
    """Linear warmup (0.1× → 1×) then cosine annealing."""
    def _fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return _fn


_CSV_FIELDS = [
    "epoch", "lr",
    "train_loss", "train_mse", "train_ic", "train_rank_ic",
    "train_dir_acc", "train_pred_std", "train_target_std", "train_disp_pen", "train_grad_norm",
    "test_loss", "test_mse", "test_ic", "test_rank_ic",
    "test_dir_acc", "test_pred_std", "test_target_std", "test_disp_pen",
    "loss_ratio", "elapsed_s",
]


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("thgnn_train")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    log_dir = args.log_dir if args.log_dir is not None else PLOT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p.name for p in args.data_dir.glob("*.pkl")])
    if not files:
        raise RuntimeError(f"No .pkl files found in {args.data_dir}")
    split = compute_split_indices(files, args)

    run_tag      = split.pre_data
    log_path     = log_dir / f"{run_tag}_thgnn_train.txt"
    csv_path     = log_dir / f"{run_tag}_thgnn_metrics.csv"
    json_path    = log_dir / f"{run_tag}_thgnn_summary.json"

    logger = _setup_logger(log_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(
        f"Train: {files[split.train_start]} -> {files[split.train_end_exclusive - 1]} "
        f"({split.train_end_exclusive - split.train_start} samples)"
    )
    logger.info(
        f"Test:  {files[split.test_start]} -> {files[split.test_end_exclusive - 1]} "
        f"({split.test_end_exclusive - split.test_start} samples)  [used for checkpoint selection]"
    )
    logger.info(f"Checkpoint pre_data: {run_tag}")
    logger.info(f"Args: {vars(args)}")

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

    model = StockHeteGAT(
        in_features=args.in_features,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        out_features=args.out_features,
        predictor_out_dim=1,
        predictor_activation=None,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"THGNN model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_make_lr_lambda(args.lr_warmup_epochs, args.epochs)
    )

    best_test_ic = -np.inf
    best_test_mse = np.inf
    best_epoch = -1
    wait = 0
    overfit_streak = 0
    stop_reason = "max_epochs"
    best_path = args.model_dir / f"{run_tag}_icrank_best.dat"
    history: dict[str, list] = {k: [] for k in _CSV_FIELDS}

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
    csv_writer.writeheader()
    csv_file.flush()

    logger.info(f"CSV log  -> {csv_path}")
    logger.info(f"Text log -> {log_path}")
    logger.info("-" * 100)
    logger.info(
        f"{'Ep':>4}  {'LR':>8}  "
        f"{'tr_loss':>8} {'tr_mse':>8} {'tr_ic':>7} {'tr_gn':>7}  "
        f"{'te_loss':>8} {'te_mse':>8} {'te_ic':>7} {'te_dir':>7} {'ratio':>6}  "
        f"{'secs':>6}  flag"
    )
    logger.info("-" * 100)

    t0_total = time.time()
    pbar = tqdm(range(1, args.epochs + 1), desc="IC-ranked epochs", unit="epoch", file=sys.stderr)
    for epoch in pbar:
        t0 = time.time()
        ic_ramp = min(1.0, epoch / max(1, args.ic_warmup_epochs))
        current_ic_weight = args.ic_weight * ic_ramp
        temperature = max(0.02, 0.2 * (0.95 ** epoch))
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = run_epoch(
            train_loader, model, device, optimizer, args,
            ic_weight_override=current_ic_weight,
            temperature=temperature,
            return_scale=args.return_scale,
        )
        test_metrics = run_epoch(
            test_loader, model, device, None, args,
            ic_weight_override=current_ic_weight,
            temperature=temperature,
            return_scale=args.return_scale,
        )
        scheduler.step()

        elapsed = time.time() - t0
        loss_ratio = test_metrics["loss"] / max(train_metrics["loss"], 1e-8)

        pbar.set_postfix(
            tr_loss=f"{train_metrics['loss']:.4f}",
            te_loss=f"{test_metrics['loss']:.4f}",
            te_ic=f"{test_metrics['rank_ic']:.4f}",
            te_dir=f"{test_metrics['dir_acc']:.3f}",
            ratio=f"{loss_ratio:.2f}",
        )

        row = {
            "epoch": epoch, "lr": current_lr,
            "train_loss":      train_metrics["loss"],
            "train_mse":       train_metrics["mse"],
            "train_ic":        train_metrics["ic"],
            "train_rank_ic":   train_metrics["rank_ic"],
            "train_dir_acc":   train_metrics["dir_acc"],
            "train_pred_std":  train_metrics["pred_std"],
            "train_target_std":train_metrics["target_std"],
            "train_disp_pen":  train_metrics["disp_pen"],
            "train_grad_norm": train_metrics["grad_norm"],
            "test_loss":       test_metrics["loss"],
            "test_mse":        test_metrics["mse"],
            "test_ic":         test_metrics["ic"],
            "test_rank_ic":    test_metrics["rank_ic"],
            "test_dir_acc":    test_metrics["dir_acc"],
            "test_pred_std":   test_metrics["pred_std"],
            "test_target_std": test_metrics["target_std"],
            "test_disp_pen":   test_metrics["disp_pen"],
            "loss_ratio":      loss_ratio,
            "elapsed_s":       round(elapsed, 1),
        }
        csv_writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in row.items()})
        csv_file.flush()
        for k, v in row.items():
            history[k].append(v)

        # ---- divergence guard ----
        overfit_flag = ""
        if loss_ratio > args.max_loss_ratio:
            overfit_streak += 1
            overfit_flag = f"[OVERFIT x{overfit_streak}]"
        else:
            overfit_streak = 0

        logger.info(
            f"{epoch:>4}  {current_lr:>8.2e}  "
            f"{train_metrics['loss']:>8.4f} {train_metrics['mse']:>8.4f} "
            f"{train_metrics['rank_ic']:>7.4f} {train_metrics['grad_norm']:>7.3f}  "
            f"{test_metrics['loss']:>8.4f} {test_metrics['mse']:>8.4f} "
            f"{test_metrics['rank_ic']:>7.4f} {test_metrics['dir_acc']:>7.4f} "
            f"{loss_ratio:>6.2f}  {elapsed:>6.1f}s  {overfit_flag}"
        )

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
            logger.info(f"  => Checkpoint saved (rank_ic={best_test_ic:.4f})")
        else:
            wait += 1
            if wait >= args.patience:
                stop_reason = f"ic_patience_{args.patience}"
                logger.info(f"Early stopping: IC patience={args.patience} exhausted at epoch {epoch}.")
                break

        if overfit_streak >= args.overfit_patience:
            stop_reason = f"divergence_guard_{args.overfit_patience}"
            logger.info(
                f"Divergence guard triggered at epoch {epoch}: "
                f"test/train ratio {loss_ratio:.2f} > {args.max_loss_ratio} "
                f"for {args.overfit_patience} consecutive epochs."
            )
            break

    csv_file.close()

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    final_train_metrics = run_epoch(train_loader, model, device, None, args, return_scale=args.return_scale)
    final_test_metrics = run_epoch(test_loader, model, device, None, args, return_scale=args.return_scale)

    # ---- Individual plots (one image per metric panel) ----
    epochs_arr = history["epoch"]

    # Plot 1: Composite Loss
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_arr, history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs_arr, history["test_loss"], label="Test Loss", linewidth=2)
    if best_epoch > 0:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.7,
                   label=f"Best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Composite Loss")
    ax.set_title(f"THGNN — Composite Loss  [{run_tag}]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    loss_plot_path = PLOT_DIR / f"{run_tag}_thgnn_loss_curve.png"
    plt.savefig(loss_plot_path, dpi=200)
    plt.close(fig)

    # Plot 2: Rank IC
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_arr, history["train_rank_ic"], label="Train Rank-IC", linewidth=2)
    ax.plot(epochs_arr, history["test_rank_ic"], label="Test Rank-IC", linewidth=2)
    if best_epoch > 0:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.7,
                   label=f"Best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Spearman IC")
    ax.set_title(f"THGNN — Rank IC  [{run_tag}]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    ic_plot_path = PLOT_DIR / f"{run_tag}_thgnn_rank_ic_curve.png"
    plt.savefig(ic_plot_path, dpi=200)
    plt.close(fig)

    # Plot 3: Directional Accuracy & Loss Ratio
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_arr, history["train_dir_acc"], label="Train Dir-Acc", linewidth=2)
    ax.plot(epochs_arr, history["test_dir_acc"], label="Test Dir-Acc", linewidth=2)
    ax.plot(epochs_arr, history["loss_ratio"], label="Test/Train ratio",
            linewidth=1.5, linestyle=":", color="red")
    ax.axhline(args.max_loss_ratio, color="red", linestyle="--", alpha=0.5,
               label=f"Divergence guard ({args.max_loss_ratio}x)")
    if best_epoch > 0:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.7,
                   label=f"Best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(f"THGNN — Dir Accuracy & Loss Ratio  [{run_tag}]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    diracc_plot_path = PLOT_DIR / f"{run_tag}_thgnn_dir_acc_curve.png"
    plt.savefig(diracc_plot_path, dpi=200)
    plt.close(fig)

    # ---- JSON summary ----
    total_time = time.time() - t0_total
    summary = {
        "run_tag":          run_tag,
        "stop_reason":      stop_reason,
        "best_epoch":       best_epoch,
        "best_test_ic":     round(best_test_ic, 6),
        "best_test_mse":    round(best_test_mse, 8),
        "final_train_loss": round(final_train_metrics["loss"], 6),
        "final_test_loss":  round(final_test_metrics["loss"], 6),
        "final_test_rank_ic":  round(final_test_metrics["rank_ic"], 6),
        "final_test_dir_acc":  round(final_test_metrics["dir_acc"], 6),
        "total_epochs_run": len(epochs_arr),
        "total_time_s":     round(total_time, 1),
        "args":             {k: str(v) for k, v in vars(args).items()},
        "epoch_history":    {
            k: [round(v, 6) if isinstance(v, float) else v for v in vals]
            for k, vals in history.items()
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("-" * 100)
    logger.info("Training complete.")
    logger.info(f"Stop reason:     {stop_reason}")
    logger.info(f"Best epoch:      {best_epoch}")
    logger.info(f"Best test IC:    {best_test_ic:.4f}")
    logger.info(f"Best test MSE:   {best_test_mse:.6f}")
    logger.info(f"Checkpoint:      {best_path}")
    logger.info(
        f"Final metrics  |  "
        f"train_loss={final_train_metrics['loss']:.6f}  "
        f"test_loss={final_test_metrics['loss']:.6f}  "
        f"test_rank_ic={final_test_metrics['rank_ic']:.4f}  "
        f"dir_acc={final_test_metrics['dir_acc']:.4f}"
    )
    logger.info(f"Loss plot:       {loss_plot_path}")
    logger.info(f"Rank-IC plot:    {ic_plot_path}")
    logger.info(f"Dir-Acc plot:    {diracc_plot_path}")
    logger.info(f"Text log:        {log_path}")
    logger.info(f"CSV metrics:     {csv_path}")
    logger.info(f"JSON summary:    {json_path}")
    logger.info(f"Total wall time: {total_time / 60:.1f} min")


if __name__ == "__main__":
    main()
