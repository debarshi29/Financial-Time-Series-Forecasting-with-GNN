"""
walk_forward_train.py — Walk-forward expanding-window retraining and evaluation.

Trains the model on progressively larger historical windows, evaluating each fold
on a held-out test period. This gives a realistic picture of how the model performs
across different market regimes rather than a single lucky/unlucky test window.

Default fold schedule (expanding train, fixed 3-month val + 6-month test):

  Fold | Train window          | Val              | Test
  -----|----------------------|-----------------|-------------------
    1  | 2015-01 → 2022-06   | 2022-07–2022-09 | 2022-10–2023-03
    2  | 2015-01 → 2023-06   | 2023-07–2023-09 | 2023-10–2024-03
    3  | 2015-01 → 2024-06   | 2024-07–2024-09 | 2024-10–2025-03
    4  | 2015-01 → 2025-06   | 2025-07–2025-09 | 2025-10–2026-03  ← live

The most recent fold's checkpoint is used for live predictions (plot_live_predictions.py).

Usage
-----
    # Run all folds (skips folds whose checkpoint already exists)
    python walk_forward_train.py

    # Evaluate only — no training, just reload checkpoints and compute test metrics
    python walk_forward_train.py --eval-only

    # Run a single fold by 0-based index
    python walk_forward_train.py --fold 3

    # Custom train-start or epochs
    python walk_forward_train.py --train-start-date 2015-01-01 --epochs 60 --patience 15
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_loader import AllGraphDataSampler
from model.Thgnn import StockHeteGAT
from train_ic_ranked import run_epoch


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "data_train_predict"
MODEL_DIR = DATA_DIR / "model_saved"
PLOT_DIR = DATA_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fold definitions
# ---------------------------------------------------------------------------

@dataclass
class Fold:
    label: str        # Human-readable name shown in plots/tables
    train_start: str  # First training date (usually fixed at 2015-01-01)
    train_end: str    # Last training date  — expanding each fold
    val_start: str    # Validation start   — for early stopping
    val_end: str      # Validation end
    test_start: str   # Test start         — held-out evaluation
    test_end: str     # Test end
    checkpoint: Path = field(default=None, init=False)  # resolved after training


def build_default_folds(train_start: str = "2015-01-01") -> list[Fold]:
    """Semi-annual expanding folds; last fold covers live 2026 predictions."""
    return [
        Fold("2022-H2", train_start, "2022-06-30", "2022-07-01", "2022-09-30", "2022-10-01", "2023-03-31"),
        Fold("2023-H2", train_start, "2023-06-30", "2023-07-01", "2023-09-30", "2023-10-01", "2024-03-31"),
        Fold("2024-H2", train_start, "2024-06-30", "2024-07-01", "2024-09-30", "2024-10-01", "2025-03-31"),
        Fold("2025-H2 (Live)", train_start, "2025-06-30", "2025-07-01", "2025-09-30", "2025-10-01", "2026-03-21"),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward expanding-window retraining.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train-start-date", type=str, default="2015-01-01",
                        help="Fixed start of training window for all folds.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--ic-weight", type=float, default=0.35)
    parser.add_argument("--dispersion-weight", type=float, default=0.2)
    parser.add_argument("--min-dispersion-ratio", type=float, default=0.2)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--in-features", type=int, default=12,
                        help="Number of input features per stock per timestep.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run only this fold (0-based index). Default: all folds.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load existing checkpoints and evaluate only.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _find_checkpoint_for_fold(fold: Fold) -> Path | None:
    """Find an existing checkpoint whose date is within 7 days of fold.train_end."""
    train_end_dt = pd.to_datetime(fold.train_end)
    for p in MODEL_DIR.glob("*_icrank_best.dat"):
        try:
            ckpt_date = pd.to_datetime(p.stem.replace("_icrank_best", ""))
            if abs((ckpt_date - train_end_dt).days) <= 7:
                return p
        except Exception:
            continue
    return None


def _latest_checkpoint() -> Path | None:
    candidates = sorted(MODEL_DIR.glob("*_icrank_best.dat"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fold(fold: Fold, args: argparse.Namespace) -> Path:
    """Run train_ic_ranked.py for this fold via subprocess. Returns checkpoint path."""
    cmd = [
        sys.executable, "train_ic_ranked.py",
        "--train-start-date", fold.train_start,
        "--train-end-date",   fold.train_end,
        "--val-start-date",   fold.val_start,
        "--val-end-date",     fold.val_end,
        "--test-start-date",  fold.test_start,
        "--test-end-date",    fold.test_end,
        "--epochs",           str(args.epochs),
        "--patience",         str(args.patience),
        "--ic-weight",        str(args.ic_weight),
        "--dispersion-weight", str(args.dispersion_weight),
        "--min-dispersion-ratio", str(args.min_dispersion_ratio),
        "--mse-weight",       str(args.mse_weight),
        "--hidden-dim",       str(args.hidden_dim),
        "--num-heads",        str(args.num_heads),
        "--dropout",          str(args.dropout),
        "--in-features",      str(args.in_features),
    ]
    print(f"\n  Command: {' '.join(cmd)}")
    before = set(MODEL_DIR.glob("*_icrank_best.dat"))
    subprocess.run(cmd, check=True, cwd=BASE_DIR)

    # Find the checkpoint created by this run
    after = set(MODEL_DIR.glob("*_icrank_best.dat"))
    new_ckpts = after - before
    if new_ckpts:
        return max(new_ckpts, key=lambda p: p.stat().st_mtime)
    # Fallback: find by date proximity
    ckpt = _find_checkpoint_for_fold(fold)
    if ckpt:
        return ckpt
    raise FileNotFoundError(
        f"Could not locate checkpoint for fold '{fold.label}' after training. "
        f"Expected a file near {fold.train_end}_icrank_best.dat in {MODEL_DIR}"
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _resolve_file_indices(data_files: list[str], start_date: str, end_date: str) -> tuple[int, int]:
    file_dates = [pd.to_datetime(Path(n).stem).normalize() for n in data_files]
    start_dt = pd.to_datetime(start_date).normalize()
    end_dt = pd.to_datetime(end_date).normalize()

    start_idx = next((i for i, d in enumerate(file_dates) if d >= start_dt), len(file_dates) - 1)
    end_idx = next((i for i, d in enumerate(file_dates) if d > end_dt), len(file_dates))
    return start_idx, end_idx


def evaluate_fold(fold: Fold, checkpoint_path: Path, device: torch.device) -> dict:
    """Load checkpoint and compute val + test metrics without retraining."""
    ckpt = _load_checkpoint(checkpoint_path, device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    # Build eval args from checkpoint config (so hyperparameters always match)
    eval_args = types.SimpleNamespace(
        target_horizon=int(cfg.get("target_horizon", 0)),
        mse_weight=float(cfg.get("mse_weight", 1.0)),
        ic_weight=float(cfg.get("ic_weight", 0.35)),
        dispersion_weight=float(cfg.get("dispersion_weight", 0.2)),
        min_dispersion_ratio=float(cfg.get("min_dispersion_ratio", 0.2)),
    )

    model = StockHeteGAT(
        in_features=int(cfg.get("in_features", 12)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        num_heads=int(cfg.get("num_heads", 4)),
        num_layers=int(cfg.get("num_layers", 1)),
        out_features=int(cfg.get("out_features", 32)),
        predictor_out_dim=state_dict["predictor.0.weight"].shape[0],
        predictor_activation=None,
        dropout=float(cfg.get("dropout", 0.3)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    data_files = sorted([p.name for p in TRAIN_DATA_DIR.glob("*.pkl")])
    if not data_files:
        raise RuntimeError(f"No pkl files found in {TRAIN_DATA_DIR}")

    def _make_loader(start_date: str, end_date: str) -> DataLoader | None:
        si, ei = _resolve_file_indices(data_files, start_date, end_date)
        ei = min(ei, len(data_files))
        if ei <= si:
            return None
        ds = AllGraphDataSampler(
            str(TRAIN_DATA_DIR), mode="val",
            data_start=0, data_middle=si, data_end=ei,
        )
        if len(ds) == 0:
            return None
        return DataLoader(ds, batch_size=1, pin_memory=False, collate_fn=lambda x: x)

    metrics: dict[str, object] = {
        "fold": fold.label,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        "val_ic_ckpt": float(ckpt.get("val_ic", float("nan"))) if isinstance(ckpt, dict) else float("nan"),
        "val_mse_ckpt": float(ckpt.get("val_mse", float("nan"))) if isinstance(ckpt, dict) else float("nan"),
    }

    # Val metrics (re-computed for consistency)
    val_loader = _make_loader(fold.val_start, fold.val_end)
    if val_loader:
        vm = run_epoch(val_loader, model, device, None, eval_args)
        metrics.update({f"val_{k}": v for k, v in vm.items()})
    else:
        print(f"  Warning: val set empty for fold '{fold.label}'")

    # Test metrics
    test_loader = _make_loader(fold.test_start, fold.test_end)
    if test_loader:
        tm = run_epoch(test_loader, model, device, None, eval_args)
        metrics.update({f"test_{k}": v for k, v in tm.items()})
        print(
            f"  Val  IC={metrics.get('val_ic', float('nan')):.4f}  "
            f"MSE={metrics.get('val_mse', float('nan')):.6f}  "
            f"dir={metrics.get('val_dir_acc', float('nan')):.3f}"
        )
        print(
            f"  Test IC={metrics.get('test_ic', float('nan')):.4f}  "
            f"MSE={metrics.get('test_mse', float('nan')):.6f}  "
            f"dir={metrics.get('test_dir_acc', float('nan')):.3f}"
        )
    else:
        print(f"  Warning: test set empty for fold '{fold.label}'")

    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("WALK-FORWARD SUMMARY")
    print("=" * 90)
    header = f"{'Fold':<20} {'Train→':<12} {'TestStart':<12} {'ValIC':>7} {'TestIC':>7} {'TestMSE':>10} {'TestDir':>8} {'Epoch':>6}"
    print(header)
    print("-" * 90)
    for r in results:
        print(
            f"{r.get('fold','?'):<20} "
            f"{r.get('train_end','?'):<12} "
            f"{r.get('test_start','?'):<12} "
            f"{r.get('val_ic', float('nan')):>7.4f} "
            f"{r.get('test_ic', float('nan')):>7.4f} "
            f"{r.get('test_mse', float('nan')):>10.6f} "
            f"{r.get('test_dir_acc', float('nan')):>8.3f} "
            f"{r.get('best_epoch', -1):>6}"
        )
    print("=" * 90)

    test_ics = [r.get("test_ic", float("nan")) for r in results]
    valid_ics = [v for v in test_ics if not np.isnan(v)]
    if valid_ics:
        print(f"\nTest IC  — mean: {np.mean(valid_ics):.4f}  std: {np.std(valid_ics):.4f}  "
              f"min: {np.min(valid_ics):.4f}  max: {np.max(valid_ics):.4f}")


def plot_results(results: list[dict]) -> None:
    if not results:
        return

    labels = [r.get("fold", "?") for r in results]
    val_ics = [r.get("val_ic", float("nan")) for r in results]
    test_ics = [r.get("test_ic", float("nan")) for r in results]
    test_mses = [r.get("test_mse", float("nan")) for r in results]
    test_dirs = [r.get("test_dir_acc", float("nan")) for r in results]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    # IC comparison
    ax = axes[0]
    w = 0.35
    ax.bar(x - w / 2, val_ics, width=w, label="Val IC", color="steelblue", alpha=0.85)
    ax.bar(x + w / 2, test_ics, width=w, label="Test IC", color="coral", alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Spearman IC")
    ax.set_title("Walk-Forward: Validation vs Test IC per Fold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for xi, v in zip(x + w / 2, test_ics):
        if not np.isnan(v):
            ax.text(xi, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # Test MSE
    ax = axes[1]
    ax.bar(x, test_mses, color="orchid", alpha=0.85)
    ax.set_ylabel("Test MSE")
    ax.set_title("Walk-Forward: Test MSE per Fold")
    ax.grid(True, alpha=0.3, axis="y")
    for xi, v in zip(x, test_mses):
        if not np.isnan(v):
            ax.text(xi, v + max(test_mses) * 0.01, f"{v:.5f}", ha="center", va="bottom", fontsize=8)

    # Test directional accuracy
    ax = axes[2]
    ax.bar(x, [d * 100 for d in test_dirs], color="mediumseagreen", alpha=0.85)
    ax.axhline(50, color="gray", linewidth=1.2, linestyle="--", label="50% (random)")
    ax.set_ylabel("Sign Accuracy (%)")
    ax.set_title("Walk-Forward: Test Directional Accuracy per Fold")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for xi, v in zip(x, test_dirs):
        if not np.isnan(v):
            ax.text(xi, v * 100 + 0.5, f"{v:.1%}", ha="center", va="bottom", fontsize=8)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=15, ha="right")
    plt.tight_layout()

    save_path = PLOT_DIR / "walk_forward_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved results chart: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    folds = build_default_folds(train_start=args.train_start_date)
    if args.fold is not None:
        if args.fold < 0 or args.fold >= len(folds):
            raise ValueError(f"--fold must be 0–{len(folds) - 1}, got {args.fold}")
        folds = [folds[args.fold]]

    results = []
    for i, fold in enumerate(folds):
        print(f"\n{'=' * 60}")
        print(f"FOLD {i + 1}/{len(folds)}: {fold.label}")
        print(f"  Train : {fold.train_start} → {fold.train_end}")
        print(f"  Val   : {fold.val_start} → {fold.val_end}")
        print(f"  Test  : {fold.test_start} → {fold.test_end}")
        print(f"{'=' * 60}")

        # Locate or train checkpoint
        existing = _find_checkpoint_for_fold(fold)

        if args.eval_only:
            if existing is None:
                print(f"  Skipping fold '{fold.label}': no checkpoint found and --eval-only is set.")
                continue
            checkpoint_path = existing
            print(f"  Using existing checkpoint: {checkpoint_path}")
        elif existing is not None:
            print(f"  Checkpoint already exists ({existing.name}), skipping training.")
            checkpoint_path = existing
        else:
            print(f"  Training fold '{fold.label}'...")
            checkpoint_path = train_fold(fold, args)
            print(f"  Checkpoint saved: {checkpoint_path}")

        fold.checkpoint = checkpoint_path

        print(f"  Evaluating on val + test sets...")
        fold_metrics = evaluate_fold(fold, checkpoint_path, device)
        results.append(fold_metrics)

    if not results:
        print("No folds evaluated.")
        return

    print_summary(results)
    plot_results(results)

    # Point to the most recent checkpoint for live predictions
    live_fold_metrics = results[-1]
    print(f"\nMost recent checkpoint (use for live predictions):")
    print(f"  {live_fold_metrics['checkpoint']}")
    print(f"\nTo generate March 2026 live plots:")
    print(
        f"  python plot_live_predictions.py "
        f"--start-date 2026-03-01 --end-date 2026-03-21 "
        f"--checkpoint {live_fold_metrics['checkpoint']}"
    )


if __name__ == "__main__":
    main()
