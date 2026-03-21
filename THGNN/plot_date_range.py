"""Plot actual vs predicted returns for the top IC-ranked stocks in a date range.

Usage examples
--------------
# Default range:
    python plot_date_range.py

# Custom range, top 5 stocks by IC:
    python plot_date_range.py --start-date 2026-03-03 --end-date 2026-03-13 --top-n 5

# Specific checkpoint:
    python plot_date_range.py --start-date 2026-01-06 --end-date 2026-01-10 --checkpoint data/model_saved/2023-12-29_icrank_best.dat
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_loader import AllGraphDataSampler
from model.Thgnn import StockHeteGAT
from trainer.trainer import extract_data


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "data_train_predict"
DAILY_STOCK_DIR = DATA_DIR / "daily_stock"
MODEL_DIR = DATA_DIR / "model_saved"
PLOT_DIR = DATA_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot actual vs predicted returns for top IC-ranked stocks in a date range."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2026-03-03",
        help="Start date (inclusive), format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-03-07",
        help="End date (inclusive), format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top IC-ranked stocks to plot.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum number of samples required per stock for IC ranking.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to a specific .dat checkpoint. Defaults to the latest *_icrank_best.dat.",
    )
    return parser.parse_args()


def _index_for_date(file_dates: list[pd.Timestamp], date_str: str) -> int:
    dt = pd.to_datetime(date_str).normalize()
    for idx, file_dt in enumerate(file_dates):
        if file_dt >= dt:
            return idx
    return len(file_dates) - 1


def _resolve_checkpoint(checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        path = Path(checkpoint_arg)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    candidates = sorted(MODEL_DIR.glob("*_icrank_best.dat"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No *_icrank_best.dat found in {MODEL_DIR}")
    return candidates[-1]


def _load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _spearman_ic(pred: pd.Series, actual: pd.Series) -> float:
    if len(pred) < 2:
        return float("nan")
    return float(pred.corr(actual, method="spearman"))


def _pearson_ic(pred: pd.Series, actual: pd.Series) -> float:
    if len(pred) < 2:
        return float("nan")
    return float(pred.corr(actual, method="pearson"))


def _plot_stock_series(
    code: str,
    stock_data: pd.DataFrame,
    row: pd.Series,
    date_range_tag: str,
) -> None:
    """Create one actual-vs-predicted time-series plot for a stock."""
    stock_data = stock_data.sort_values("dt").copy()
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        stock_data["dt"],
        stock_data["actual_return"],
        color="black",
        linewidth=2,
        marker="o",
        markersize=5,
        label="Actual Return",
    )
    ax.plot(
        stock_data["dt"],
        stock_data["predicted_return"],
        color="royalblue",
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=5,
        label="Predicted Return",
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title(f"{code} [{date_range_tag}]")
    ax.set_ylabel("Return")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.text(
        0.01,
        0.98,
        (
            f"Spearman IC: {row['spearman_ic']:.4f}\n"
            f"Pearson IC:  {row['pearson_ic']:.4f}\n"
            f"Sign Acc:    {row['sign_acc']:.2%}\n"
            f"MSE:         {row['mse']:.6f}\n"
            f"Samples:     {int(row['n_samples'])}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    save_path = PLOT_DIR / f"stock_{date_range_tag}_{code}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_top_stock_grid(top_n: pd.DataFrame, df_results: pd.DataFrame, date_range_tag: str) -> None:
    """Create one combined figure with one subplot per IC-ranked stock."""
    if top_n.empty:
        return

    n_rows = len(top_n)
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4.5 * n_rows), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, (_, row) in zip(axes, top_n.iterrows()):
        code = row["code"]
        stock_data = df_results[df_results["code"] == code].sort_values("dt")

        ax.plot(
            stock_data["dt"],
            stock_data["actual_return"],
            color="black",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Actual Return",
        )
        ax.plot(
            stock_data["dt"],
            stock_data["predicted_return"],
            color="royalblue",
            linewidth=2,
            linestyle="--",
            marker="s",
            markersize=4,
            label="Predicted Return",
        )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(
            f"{code} | Spearman IC: {row['spearman_ic']:.4f} | "
            f"Pearson IC: {row['pearson_ic']:.4f} | MSE: {row['mse']:.6f}"
        )
        ax.set_ylabel("Return")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    fig.suptitle(f"Top {len(top_n)} Stocks by Spearman IC [{date_range_tag}]", fontsize=14, y=0.995)
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    save_path = PLOT_DIR / f"top_{len(top_n)}_ic_{date_range_tag}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined top-stock plot: {save_path}")


def _collect_predictions(
    dataset: AllGraphDataSampler,
    model: StockHeteGAT,
    device: torch.device,
) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    print("Running predictions...")

    for i in tqdm(range(len(dataset)), unit="day"):
        pkl_name = dataset.gnames_all[i]
        date_str = Path(pkl_name).stem
        daily_file = DAILY_STOCK_DIR / f"{date_str}.csv"
        if not daily_file.exists():
            continue

        df_meta = pd.read_csv(daily_file, dtype=object)
        sample = dataset[i]
        pos_adj, neg_adj, features, labels, mask = extract_data(sample, str(device))

        with torch.no_grad():
            logits = model(features, pos_adj, neg_adj)

        preds = logits.detach().cpu().numpy()
        lbls = labels.detach().cpu().numpy()

        if preds.ndim == 1:
            preds = preds[:, None]
        if lbls.ndim == 1:
            lbls = lbls[:, None]

        for idx, row in df_meta.iterrows():
            if idx >= len(preds):
                break
            results.append(
                {
                    "dt": pd.to_datetime(row["dt"]),
                    "code": row["code"],
                    "actual_return": float(lbls[idx, 0]),
                    "predicted_return": float(preds[idx, 0]),
                }
            )

    return pd.DataFrame(results)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    date_range_tag = f"{args.start_date}_to_{args.end_date}"

    data_files = sorted([p.name for p in TRAIN_DATA_DIR.glob("*.pkl")])
    if not data_files:
        raise RuntimeError(f"No .pkl files found in {TRAIN_DATA_DIR}")

    file_dates = [pd.to_datetime(Path(name).stem).normalize() for name in data_files]
    start_idx = _index_for_date(file_dates, args.start_date)
    end_idx = _index_for_date(file_dates, args.end_date) + 1
    end_idx = min(end_idx, len(data_files))

    if end_idx <= start_idx:
        raise ValueError(
            f"No pkl files found between {args.start_date} and {args.end_date}. "
            "Run the data pipeline first to generate pkl files for that range."
        )

    actual_start = file_dates[start_idx].date()
    actual_end = file_dates[end_idx - 1].date()
    print(f"Date range : {actual_start} -> {actual_end} ({end_idx - start_idx} trading day(s))")

    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    print(f"Checkpoint : {checkpoint_path}")
    checkpoint = _load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    dataset = AllGraphDataSampler(
        str(TRAIN_DATA_DIR),
        mode="val",
        data_start=0,
        data_middle=start_idx,
        data_end=end_idx,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty for the selected date range.")

    target_dim = state_dict["predictor.0.weight"].shape[0]
    model = StockHeteGAT(
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_heads=int(cfg.get("num_heads", 8)),
        num_layers=int(cfg.get("num_layers", 1)),
        out_features=int(cfg.get("out_features", 32)),
        predictor_out_dim=target_dim,
        predictor_activation=None,
        dropout=float(cfg.get("dropout", 0.3)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    df_results = _collect_predictions(dataset, model, device)
    if df_results.empty:
        raise RuntimeError("No predictions generated. Check that pkl files exist for the date range.")

    metrics = []
    for code, group in df_results.groupby("code"):
        pred_ret = group["predicted_return"].astype(float)
        act_ret = group["actual_return"].astype(float)

        metrics.append(
            {
                "code": code,
                "spearman_ic": _spearman_ic(pred_ret, act_ret),
                "pearson_ic": _pearson_ic(pred_ret, act_ret),
                "sign_acc": float(np.mean(np.sign(pred_ret) == np.sign(act_ret))),
                "mse": float(np.mean((act_ret - pred_ret) ** 2)),
                "n_samples": len(group),
            }
        )

    df_metrics = pd.DataFrame(metrics)
    if df_metrics.empty:
        raise RuntimeError("No stock metrics computed for the selected range.")

    valid_ic = df_metrics["spearman_ic"].notna() & (df_metrics["n_samples"] >= args.min_samples)
    if not valid_ic.any():
        raise RuntimeError(
            "IC ranking requires at least 2 samples per stock in the selected date range. "
            "Choose a wider date range."
        )

    top_n = df_metrics[valid_ic].sort_values(
        by=["spearman_ic", "pearson_ic", "sign_acc", "mse"],
        ascending=[False, False, False, True],
    ).head(args.top_n)

    print(f"\nTop {len(top_n)} stocks ranked by Spearman IC:")
    print(top_n[["code", "spearman_ic", "pearson_ic", "sign_acc", "mse", "n_samples"]].to_string(index=False))

    print(f"\nGenerating combined actual-vs-predicted plot for top {len(top_n)} stocks...")
    _plot_top_stock_grid(top_n, df_results, date_range_tag)

    print(f"\nGenerating individual actual-vs-predicted plots for top {len(top_n)} stocks...")
    for _, row in top_n.iterrows():
        code = row["code"]
        stock_data = df_results[df_results["code"] == code].sort_values("dt")
        _plot_stock_series(code, stock_data, row, date_range_tag)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
