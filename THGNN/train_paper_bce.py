"""train_paper_bce.py

Training script that replicates the THGNN paper's original loss function and
evaluation protocol (Section 3.6 and Section 4.1 of the paper).

Loss (Equation 10):
    L = -Σ [Y_l · log(Ŷ_l) + (1 - Y_l) · log(1 - Ŷ_l)]
    Binary cross-entropy where Y=1 means price goes up, Y=0 means price goes down.

Evaluation metrics (Table 2):
    ACC  – Prediction accuracy (directional: does sign(pred) match sign(return)?)
    ARR  – Annualised Rate of Return of the long-top-k portfolio
    AV   – Annualised Volatility of daily portfolio returns (lower → less risky)
    MDD  – Maximum Draw Down of the portfolio equity curve (lower abs → less risky)
    ASR  – Annual Sharpe Ratio = ARR / AV
    CR   – Calmar Ratio = ARR / |MDD|
    IR   – Information Ratio = annualised(alpha) / annualised_vol(alpha),
           where alpha = daily portfolio return − daily equal-weight benchmark return

Everything else (model, data loading, optimiser, scheduler, split logic) is
identical to train_ic_ranked.py so results can be compared directly.
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
    parser = argparse.ArgumentParser(
        description="Train THGNN with paper's binary cross-entropy loss and evaluate "
                    "with ACC, ARR, AV, MDD, ASR, CR, IR as in Table 2 of the paper."
    )
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
                        help="Number of input features per stock per timestep.")
    parser.add_argument("--target-horizon", type=int, default=0,
                        help="Which label horizon to train on (0=next-day). "
                             "Mirrors the same argument in train_ic_ranked.py.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top-ranked stocks to hold long each day when "
                             "computing portfolio metrics ARR/AV/MDD/ASR/CR/IR. "
                             "The paper uses top-100 out of 500 stocks; with Nifty50 "
                             "the default top-5 is an equivalent 10%% slice.")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── split-index helpers (identical to train_ic_ranked.py) ─────────────────────

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


# ── paper loss (Equation 10) ───────────────────────────────────────────────────

def bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Binary cross-entropy loss as in the THGNN paper (Equation 10).

    Ŷ = σ(WZ + b)
    L = -Σ [Y_l · log(Ŷ_l) + (1 - Y_l) · log(1 - Ŷ_l)]

    Continuous return labels are converted to binary here:
        Y = 1  if return > 0  (price goes up)
        Y = 0  if return ≤ 0  (price stays flat or goes down)

    BCEWithLogitsLoss is used instead of explicit sigmoid + BCE for numerical
    stability — mathematically equivalent to the paper's formulation.
    """
    logits_flat = logits.reshape(-1)
    target_flat = target.reshape(-1)

    # Binary labels as in the paper (up=1, down/flat=0)
    binary_labels = (target_flat > 0).float()

    loss = F.binary_cross_entropy_with_logits(logits_flat, binary_labels)

    with torch.no_grad():
        probs = torch.sigmoid(logits_flat)
        pred_labels = (probs > 0.5).float()
        acc = (pred_labels == binary_labels).float().mean()

    metrics = {
        "bce": float(loss.detach().cpu().item()),
        "acc": float(acc.cpu().item()),
    }
    return loss, metrics


# ── portfolio metrics (Table 2) ────────────────────────────────────────────────

def compute_portfolio_metrics(
    portfolio_days: list[tuple[np.ndarray, np.ndarray]],
    top_k: int,
    trading_days_per_year: int = 252,
) -> dict[str, float]:
    """Compute all 7 paper metrics from per-day (predicted_probs, actual_returns).

    Trading strategy mirrors the paper (Section 4.1):
        • Each day, rank stocks by predicted probability (descending).
        • Hold the top-k stocks with equal weight.
        • Sell at open of day t+1 (daily rebalance).

    Benchmark for IR: equal-weight portfolio of all stocks (market proxy).

    Args:
        portfolio_days: list of (probs_array, returns_array) one entry per test day.
        top_k: number of stocks to hold long each day.
        trading_days_per_year: annualisation factor (252 for equity markets).

    Returns:
        dict with keys acc, arr, av, mdd, asr, cr, ir.
    """
    if not portfolio_days:
        return {k: 0.0 for k in ["acc", "arr", "av", "mdd", "asr", "cr", "ir"]}

    daily_port_ret: list[float] = []
    daily_bench_ret: list[float] = []
    total_correct = 0
    total_stocks = 0

    for probs, returns in portfolio_days:
        n = len(probs)
        k = min(top_k, n)

        # ACC: directional accuracy across all stocks on this day
        pred_dir = (probs > 0.5).astype(float)
        actual_dir = (returns > 0).astype(float)
        total_correct += int(np.sum(pred_dir == actual_dir))
        total_stocks += n

        # Long top-k by predicted probability (equal-weight)
        top_idx = np.argsort(probs)[-k:]
        daily_port_ret.append(float(returns[top_idx].mean()))

        # Equal-weight benchmark (all stocks, market proxy)
        daily_bench_ret.append(float(returns.mean()))

    acc = total_correct / max(total_stocks, 1)

    r = np.array(daily_port_ret)
    b = np.array(daily_bench_ret)
    T = len(r)

    # ARR: Annualised Rate of Return via geometric compounding
    cum_return = float(np.prod(1.0 + r))
    arr = cum_return ** (trading_days_per_year / T) - 1.0

    # AV: Annualised Volatility
    av = float(r.std(ddof=1) * np.sqrt(trading_days_per_year)) if T > 1 else 0.0

    # MDD: Maximum Draw Down
    cumulative = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(cumulative)
    # Avoid division by zero on flat equity curves
    safe_max = np.where(running_max > 0, running_max, 1.0)
    drawdowns = (running_max - cumulative) / safe_max
    mdd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    # ASR: Annual Sharpe Ratio (risk-free rate = 0 to match paper)
    asr = arr / av if av > 1e-8 else 0.0

    # CR: Calmar Ratio
    cr = arr / mdd if mdd > 1e-8 else 0.0

    # IR: Information Ratio vs equal-weight benchmark
    alpha = r - b
    ann_alpha = float(alpha.mean()) * trading_days_per_year
    ann_te = float(alpha.std(ddof=1)) * np.sqrt(trading_days_per_year) if T > 1 else 1e-8
    ir = ann_alpha / ann_te if ann_te > 1e-8 else 0.0

    return {
        "acc": float(acc),
        "arr": float(arr),
        "av": float(av),
        "mdd": float(mdd),
        "asr": float(asr),
        "cr": float(cr),
        "ir": float(ir),
    }


# ── training loop ──────────────────────────────────────────────────────────────

def run_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[tuple[np.ndarray, np.ndarray]]]:
    """Run one training or evaluation epoch.

    Returns:
        metrics: dict with averaged loss/acc over all batches.
        portfolio_days: list of (probs, returns) per graph sample.
                        Non-empty only during evaluation (optimizer=None).
    """
    training = optimizer is not None
    model.train(mode=training)

    sums: dict[str, float] = {"loss": 0.0, "bce": 0.0, "acc": 0.0}
    steps = 0
    portfolio_days: list[tuple[np.ndarray, np.ndarray]] = []

    for batch in loader:
        sample_list = batch if isinstance(batch, list) else [batch]
        for data in sample_list:
            pos_adj, neg_adj, features, labels, mask = extract_data(data, str(device))

            # Single horizon — avoids conflicting gradients from correlated label horizons
            if labels.dim() > 1 and labels.shape[-1] > 1:
                labels = labels[:, args.target_horizon]

            mask_t = _mask_to_tensor(mask, len(labels), device)
            logits = model(features, pos_adj, neg_adj)

            logits = _ensure_2d(logits)
            labels = _ensure_2d(labels)
            pred = logits[mask_t].reshape(-1)
            target = labels[mask_t].reshape(-1)

            if pred.numel() == 0 or target.numel() == 0:
                continue

            loss, metrics = bce_loss(pred, target)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            sums["loss"] += float(loss.detach().cpu().item())
            for key in ["bce", "acc"]:
                sums[key] += metrics[key]
            steps += 1

            # Collect raw probabilities and actual returns for portfolio simulation
            if not training:
                with torch.no_grad():
                    probs = torch.sigmoid(pred.detach()).cpu().numpy()
                    returns = target.detach().cpu().numpy()
                portfolio_days.append((probs, returns))

    avg = {k: v / steps for k, v in sums.items()} if steps > 0 else {k: 0.0 for k in sums}
    return avg, portfolio_days


def _make_lr_lambda(warmup_epochs: int, total_epochs: int):
    """Linear warmup (0.1× → 1×) then cosine annealing (identical to train_ic_ranked.py)."""
    def _fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return _fn


# ── main ───────────────────────────────────────────────────────────────────────

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

    # Model identical to train_ic_ranked.py — output is a single logit per stock.
    # BCEWithLogitsLoss in bce_loss() applies sigmoid internally, so no activation here.
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_make_lr_lambda(warmup_epochs, args.epochs)
    )

    best_test_acc = -np.inf
    best_epoch = -1
    wait = 0
    best_path = args.model_dir / f"{split.pre_data}_bce_best.dat"
    history: dict[str, list] = {"epoch": [], "train_loss": [], "test_loss": []}

    pbar = tqdm(range(1, args.epochs + 1), desc="BCE epochs", unit="epoch")
    for epoch in pbar:
        train_metrics, _ = run_epoch(train_loader, model, device, optimizer, args)
        test_metrics, _ = run_epoch(test_loader, model, device, None, args)
        scheduler.step()

        pbar.set_postfix(
            train_bce=f"{train_metrics['bce']:.4f}",
            test_bce=f"{test_metrics['bce']:.4f}",
            test_acc=f"{test_metrics['acc']:.4f}",
        )
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["test_loss"].append(test_metrics["loss"])

        # Checkpoint on test accuracy (ACC is the paper's first reported metric)
        if test_metrics["acc"] > best_test_acc:
            best_test_acc = test_metrics["acc"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "test_acc": best_test_acc,
                    "test_bce": test_metrics["bce"],
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

    # ── final evaluation with best checkpoint ─────────────────────────────────
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])

    final_train_metrics, train_portfolio_days = run_epoch(train_loader, model, device, None, args)
    final_test_metrics, test_portfolio_days = run_epoch(test_loader, model, device, None, args)

    train_port = compute_portfolio_metrics(train_portfolio_days, top_k=args.top_k)
    test_port = compute_portfolio_metrics(test_portfolio_days, top_k=args.top_k)

    # ── loss curve plot ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_loss"], label="Train BCE Loss", linewidth=2)
    plt.plot(history["epoch"], history["test_loss"], label="Test BCE Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Paper BCE Training/Test Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_plot_path = PLOT_DIR / f"{split.pre_data}_bce_loss_curve.png"
    plt.savefig(loss_plot_path, dpi=200)
    plt.close(fig)

    # ── results summary ────────────────────────────────────────────────────────
    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}  |  Best test ACC: {best_test_acc:.4f}")
    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved loss plot: {loss_plot_path}")

    header = f"{'Metric':<8}  {'Train':>10}  {'Test':>10}  Note"
    print("\n" + "=" * 60)
    print("Paper evaluation metrics (Table 2 style)")
    print("=" * 60)
    print(header)
    print("-" * 60)
    note = {
        "acc":  "higher better",
        "arr":  "higher better",
        "av":   "lower  better",
        "mdd":  "lower  better",
        "asr":  "higher better",
        "cr":   "higher better",
        "ir":   "higher better",
    }
    for key in ["acc", "arr", "av", "mdd", "asr", "cr", "ir"]:
        tv = train_port[key]
        ev = test_port[key]
        print(f"{key.upper():<8}  {tv:>10.4f}  {ev:>10.4f}  {note[key]}")
    print("=" * 60)
    print(
        f"\nNote: ARR/AV/MDD/ASR/CR/IR computed from long top-{args.top_k} "
        "portfolio (equal-weight, daily rebalance).\n"
        "IR benchmark: equal-weight portfolio of all stocks."
    )


if __name__ == "__main__":
    main()
