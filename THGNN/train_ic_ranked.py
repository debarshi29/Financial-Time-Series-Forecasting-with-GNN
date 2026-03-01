from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class SplitIndices:
    train_start: int
    train_end_exclusive: int
    val_start: int
    val_end_exclusive: int
    pre_data: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train THGNN with MSE + IC + dispersion regularization.")
    parser.add_argument("--data-dir", type=Path, default=TRAIN_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--train-start-date", type=str, default="2015-01-01")
    parser.add_argument("--train-end-date", type=str, default="2023-12-29")
    parser.add_argument("--val-start-date", type=str, default="2024-01-01")
    parser.add_argument("--val-end-date", type=str, default="2025-12-31")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--out-features", type=int, default=32)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--ic-weight", type=float, default=0.35)
    parser.add_argument("--dispersion-weight", type=float, default=0.2)
    parser.add_argument("--min-dispersion-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def compute_split_indices(files: list[str], args: argparse.Namespace) -> SplitIndices:
    file_dates = [pd.to_datetime(Path(name).stem).normalize() for name in files]
    train_start = _index_for_date(file_dates, args.train_start_date)
    train_end_idx = _index_for_date(file_dates, args.train_end_date)
    val_start = _index_for_date(file_dates, args.val_start_date)
    val_end_exclusive = _index_for_date(file_dates, args.val_end_date) + 1
    val_end_exclusive = min(val_end_exclusive, len(files))

    # Force temporal ordering.
    if val_start <= train_start:
        raise ValueError("val_start_date must be after train_start_date.")
    if train_end_idx >= val_start:
        train_end_exclusive = val_start
    else:
        train_end_exclusive = train_end_idx + 1

    if train_end_exclusive <= train_start:
        raise ValueError("Training split is empty.")
    if val_end_exclusive <= val_start:
        raise ValueError("Validation split is empty.")

    pre_data = Path(files[train_end_exclusive - 1]).stem
    return SplitIndices(
        train_start=train_start,
        train_end_exclusive=train_end_exclusive,
        val_start=val_start,
        val_end_exclusive=val_end_exclusive,
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


def pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.reshape(-1)
    y = y.reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(torch.sum(x * x) * torch.sum(y * y) + eps)
    if denom <= eps:
        return torch.tensor(0.0, device=x.device)
    return torch.sum(x * y) / denom


def composite_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mse_weight: float,
    ic_weight: float,
    dispersion_weight: float,
    min_dispersion_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = _ensure_2d(pred)
    target = _ensure_2d(target)

    mse = F.mse_loss(pred, target)
    corr = pearson_corr(pred, target)
    ic_loss = 1.0 - corr

    pred_std = pred.reshape(-1).std(unbiased=False)
    target_std = target.reshape(-1).std(unbiased=False).detach()
    min_std = min_dispersion_ratio * target_std
    dispersion_penalty = F.relu(min_std - pred_std)

    loss = (
        mse_weight * mse
        + ic_weight * ic_loss
        + dispersion_weight * dispersion_penalty
    )
    metrics = {
        "mse": float(mse.detach().cpu().item()),
        "ic": float(corr.detach().cpu().item()),
        "pred_std": float(pred_std.detach().cpu().item()),
        "target_std": float(target_std.detach().cpu().item()),
        "disp_pen": float(dispersion_penalty.detach().cpu().item()),
    }
    return loss, metrics


def run_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(mode=training)

    sums = {"loss": 0.0, "mse": 0.0, "ic": 0.0, "pred_std": 0.0, "target_std": 0.0, "disp_pen": 0.0}
    steps = 0

    for batch in loader:
        sample_list = batch if isinstance(batch, list) else [batch]
        for data in sample_list:
            pos_adj, neg_adj, features, labels, mask = extract_data(data, str(device))
            mask_t = _mask_to_tensor(mask, len(labels), device)
            logits = model(features, pos_adj, neg_adj)

            logits = _ensure_2d(logits)
            labels = _ensure_2d(labels)
            pred = logits[mask_t]
            target = labels[mask_t]
            if pred.numel() == 0 or target.numel() == 0:
                continue

            loss, metrics = composite_loss(
                pred,
                target,
                mse_weight=args.mse_weight,
                ic_weight=args.ic_weight,
                dispersion_weight=args.dispersion_weight,
                min_dispersion_ratio=args.min_dispersion_ratio,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            sums["loss"] += float(loss.detach().cpu().item())
            for key in ["mse", "ic", "pred_std", "target_std", "disp_pen"]:
                sums[key] += metrics[key]
            steps += 1

    if steps == 0:
        return {k: 0.0 for k in sums}
    return {k: v / steps for k, v in sums.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.model_dir.mkdir(parents=True, exist_ok=True)

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
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Train/val dataset empty after split filtering.")

    train_loader = DataLoader(train_ds, batch_size=1, pin_memory=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_ds, batch_size=1, pin_memory=True, collate_fn=lambda x: x)

    _, _, _, sample_labels, _ = extract_data(train_ds[0], str(device))
    target_dim = int(sample_labels.shape[-1]) if sample_labels.dim() > 1 else 1

    model = StockHeteGAT(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        out_features=args.out_features,
        predictor_out_dim=target_dim,
        predictor_activation=None,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_ic = -np.inf
    best_val_mse = np.inf
    best_epoch = -1
    wait = 0
    best_path = args.model_dir / f"{split.pre_data}_icrank_best.dat"

    pbar = tqdm(range(1, args.epochs + 1), desc="IC-ranked epochs", unit="epoch")
    for epoch in pbar:
        train_metrics = run_epoch(train_loader, model, device, optimizer, args)
        val_metrics = run_epoch(val_loader, model, device, None, args)
        scheduler.step()

        pbar.set_postfix(
            train_mse=f"{train_metrics['mse']:.6f}",
            val_mse=f"{val_metrics['mse']:.6f}",
            val_ic=f"{val_metrics['ic']:.4f}",
            spread=f"{(val_metrics['pred_std'] / max(val_metrics['target_std'], 1e-8)):.3f}",
        )

        improved = (val_metrics["ic"] > best_val_ic) or (
            np.isclose(val_metrics["ic"], best_val_ic) and val_metrics["mse"] < best_val_mse
        )
        if improved:
            best_val_ic = val_metrics["ic"]
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_ic": best_val_ic,
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

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val IC: {best_val_ic:.4f}")
    print(f"Best val MSE: {best_val_mse:.6f}")
    print(f"Saved best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
