"""
plot_live_predictions.py — Real-time prediction and evaluation plots for the Hybrid model.

Loads the best hybrid checkpoint, runs inference on the available graph samples,
and produces four types of output:

  1. Per-stock return + price plots   (top N stocks by Spearman IC)
  2. Combined return grid             (all top N in one figure)
  3. Daily cross-sectional rankings   (bar chart per trading day)
  4. Long-short portfolio chart       (cumulative P&L of top-N long / bottom-N short)

Prerequisites
-------------
Graph pkl files and daily_stock CSVs must exist. If not, run the THGNN data pipeline:

    cd ../THGNN
    python utils/download_market_data.py --start 2015-01-01
    python utils/generate_relation.py --data-path data/nifty50.pkl --relation-dir data/relation
    python rebuild_graph_data.py --threshold 0.3

Then train the hybrid model:

    cd ../THGNN_MaGNet
    python train_hybrid.py

Usage
-----
# Default: plots most recent *_hybrid_best.dat checkpoint over a date range
python plot_live_predictions.py

# Custom range
python plot_live_predictions.py --start-date 2026-03-01 --end-date 2026-03-21 --top-n 5

# Specific checkpoint
python plot_live_predictions.py --start-date 2026-03-01 --end-date 2026-03-21 \\
    --checkpoint ../THGNN/data/model_saved/2025-09-30_hybrid_best.dat
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
from model.hybrid_model import HybridStockModel


# Data lives in the THGNN sibling directory (shared with the training script)
_HERE = Path(__file__).resolve().parent
_THGNN_DIR = _HERE.parent / "THGNN"
DATA_DIR = _THGNN_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "data_train_predict"
DAILY_STOCK_DIR = DATA_DIR / "daily_stock"
MODEL_DIR = DATA_DIR / "model_saved"
PLOT_DIR = DATA_DIR / "plots"
NIFTY_CSV_DIR = DATA_DIR / "nifty50_csv"

PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time prediction plots for the Hybrid THGNN×MaGNet model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start-date", type=str, default="2026-03-01",
                        help="First date to predict (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default="2026-03-21",
                        help="Last date to predict (YYYY-MM-DD).")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top IC-ranked stocks to plot individually.")
    parser.add_argument("--long-short-n", type=int, default=5,
                        help="Stocks on each side of the simulated long-short portfolio.")
    parser.add_argument("--min-samples", type=int, default=2,
                        help="Minimum trading days required per stock for IC ranking.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific .dat checkpoint. "
                             "Defaults to the most recently modified *_hybrid_best.dat.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _index_for_date(file_dates: list[pd.Timestamp], date_str: str) -> int:
    dt = pd.to_datetime(date_str).normalize()
    for idx, d in enumerate(file_dates):
        if d >= dt:
            return idx
    return len(file_dates) - 1


def _resolve_checkpoint(checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        path = Path(checkpoint_arg)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    candidates = sorted(MODEL_DIR.glob("*_hybrid_best.dat"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No *_hybrid_best.dat found in {MODEL_DIR}. "
            "Run train_hybrid.py first."
        )
    return candidates[-1]


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _pick_valid_num_heads(embed_dim: int, requested: int, *, fallback: int) -> int:
    if requested > 0 and embed_dim % requested == 0:
        return requested
    if fallback > 0 and embed_dim % fallback == 0:
        return fallback
    for candidate in range(min(embed_dim, max(requested, fallback, 1)), 0, -1):
        if embed_dim % candidate == 0:
            return candidate
    return 1


def _infer_model_kwargs(cfg: dict, state_dict: dict) -> dict:
    required = [
        "embed.weight",
        "predictor.0.weight",
        "pos_gat.weight",
        "gph.to_hyper.weight",
        "tch.to_hyper.2.weight",
    ]
    missing = [key for key in required if key not in state_dict]
    if missing:
        raise ValueError(
            "Checkpoint does not match HybridStockModel. "
            f"Missing key(s): {', '.join(missing)}"
        )

    embed_dim, in_features = state_dict["embed.weight"].shape
    predictor_out_dim = state_dict["predictor.0.weight"].shape[0]
    gat_weight = state_dict["pos_gat.weight"]
    gat_out_features = int(cfg.get("gat_out_features", 8))
    if gat_weight.ndim != 2 or gat_weight.shape[0] != embed_dim:
        raise ValueError(
            "Checkpoint contains unexpected GAT weight shape for HybridStockModel: "
            f"{tuple(gat_weight.shape)}"
        )
    if gat_out_features <= 0 or gat_weight.shape[1] % gat_out_features != 0:
        gat_out_features = next(
            (
                candidate
                for candidate in range(min(32, gat_weight.shape[1]), 0, -1)
                if gat_weight.shape[1] % candidate == 0
            ),
            gat_weight.shape[1],
        )
    gat_heads = gat_weight.shape[1] // gat_out_features

    num_mage_layers = len({
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("mage_layers.") and key.split(".")[1].isdigit()
    })
    if num_mage_layers == 0:
        raise ValueError("Checkpoint has no MAGE layers; not a valid hybrid checkpoint.")

    num_moe_experts = state_dict.get("mage_layers.0.moe.gate.weight", torch.empty(4, embed_dim)).shape[0]
    num_hyper_edges = state_dict["gph.to_hyper.weight"].shape[0]
    num_tch_hyper_edges = state_dict["tch.to_hyper.2.weight"].shape[0]

    requested_mha_heads = int(cfg.get("num_mha_heads", 2))
    requested_tch_heads = int(cfg.get("num_tch_heads", 4))

    return {
        "in_features": int(cfg.get("in_features", in_features)),
        "embed_dim": int(cfg.get("embed_dim", embed_dim)),
        "num_mage_layers": int(cfg.get("num_mage_layers", num_mage_layers)),
        "num_moe_experts": int(cfg.get("num_moe_experts", num_moe_experts)),
        "num_mha_heads": _pick_valid_num_heads(embed_dim, requested_mha_heads, fallback=2),
        "gat_heads": int(cfg.get("gat_heads", gat_heads)),
        "gat_out_features": int(cfg.get("gat_out_features", gat_out_features)),
        "num_hyper_edges": int(cfg.get("num_hyper_edges", num_hyper_edges)),
        "num_tch_hyper_edges": int(cfg.get("num_tch_hyper_edges", num_tch_hyper_edges)),
        "num_tch_heads": _pick_valid_num_heads(embed_dim, requested_tch_heads, fallback=4),
        "dropout": float(cfg.get("dropout", 0.1)),
        "predictor_out_dim": predictor_out_dim,
        "predictor_activation": None,
    }


def _build_model(cfg: dict, state_dict: dict, device: torch.device) -> HybridStockModel:
    """Reconstruct HybridStockModel from the config saved inside the checkpoint."""
    model_kwargs = _infer_model_kwargs(cfg, state_dict)
    try:
        model = HybridStockModel(**model_kwargs).to(device)
    except Exception as exc:
        raise RuntimeError(
            "Failed to reconstruct HybridStockModel from checkpoint with "
            f"config {model_kwargs}"
        ) from exc
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint is incompatible with the current HybridStockModel definition. "
            f"Resolved config: {model_kwargs}"
        ) from exc
    model.eval()
    return model


def _extract_data(data_dict: dict, device: str):
    """Move a sample dict to device and strip spurious batch dimensions."""
    pos_adj  = data_dict["pos_adj"].to(device)
    neg_adj  = data_dict["neg_adj"].to(device)
    features = data_dict["features"].to(device)
    labels   = data_dict["labels"].to(device)

    def _squeeze(t: torch.Tensor, min_dims: int) -> torch.Tensor:
        while t.dim() > min_dims and t.size(0) == 1:
            t = t.squeeze(0)
        return t

    features = _squeeze(features, 3)
    pos_adj  = _squeeze(pos_adj, 2)
    neg_adj  = _squeeze(neg_adj, 2)
    if labels.dim() > 1 and labels.size(-1) == 1:
        labels = labels.squeeze(-1)
    return pos_adj, neg_adj, features, labels, data_dict["mask"]


def _spearman_ic(pred: pd.Series, actual: pd.Series) -> float:
    if len(pred) < 2:
        return float("nan")
    return float(pred.corr(actual, method="spearman"))


def _collect_predictions(
    dataset: AllGraphDataSampler,
    model: HybridStockModel,
    device: torch.device,
) -> pd.DataFrame:
    results: list[dict] = []
    print("Running predictions...")
    for i in tqdm(range(len(dataset)), unit="day"):
        pkl_name = dataset.gnames_all[i]
        date_str = Path(pkl_name).stem
        daily_file = DAILY_STOCK_DIR / f"{date_str}.csv"
        if not daily_file.exists():
            continue
        df_meta = pd.read_csv(daily_file, dtype=object)
        sample = dataset[i]
        pos_adj, neg_adj, features, labels, mask = _extract_data(sample, str(device))
        with torch.no_grad():
            logits = model(features, pos_adj, neg_adj)
        preds = logits.detach().cpu().numpy()
        lbls  = labels.detach().cpu().numpy()
        if preds.ndim == 1:
            preds = preds[:, None]
        if lbls.ndim == 1:
            lbls = lbls[:, None]
        for idx, row in df_meta.iterrows():
            if idx >= len(preds):
                break
            results.append({
                "dt": pd.to_datetime(row["dt"]),
                "code": row["code"],
                "actual_return": float(lbls[idx, 0]),
                "predicted_return": float(preds[idx, 0]),
            })
    return pd.DataFrame(results)


def _load_prices(code: str) -> pd.DataFrame | None:
    csv_path = NIFTY_CSV_DIR / f"{code.replace('.', '_')}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["dt"] = pd.to_datetime(df["Date"])
    return df[["dt", "Close"]]


# ---------------------------------------------------------------------------
# Plot 1: Individual stock (returns + price)
# ---------------------------------------------------------------------------

def _plot_stock(
    code: str,
    stock_data: pd.DataFrame,
    row: pd.Series,
    price_df: pd.DataFrame | None,
    date_range_tag: str,
) -> None:
    stock_data = stock_data.sort_values("dt").copy()
    merged = None
    if price_df is not None:
        merged = pd.merge(stock_data, price_df, on="dt", how="inner")
        if merged.empty:
            merged = None

    n_panels = 2 if merged is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(stock_data["dt"], stock_data["actual_return"] * 100,
            color="black", linewidth=2, marker="o", markersize=5, label="Actual Return")
    ax.plot(stock_data["dt"], stock_data["predicted_return"] * 100,
            color="royalblue", linewidth=2, linestyle="--", marker="s", markersize=5,
            label="Predicted Return")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title(f"{code}  |  Hybrid Model Predictions  [{date_range_tag}]", fontsize=12)
    ax.set_ylabel("Return (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.text(
        0.01, 0.98,
        f"Spearman IC: {row['spearman_ic']:.4f}\n"
        f"Sign Acc:    {row['sign_acc']:.2%}\n"
        f"MSE:         {row['mse']:.6f}\n"
        f"Days:        {int(row['n_samples'])}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    if merged is not None:
        merged["actual_next_price"]    = merged["Close"] * (1 + merged["actual_return"])
        merged["predicted_next_price"] = merged["Close"] * (1 + merged["predicted_return"])
        ax2 = axes[1]
        ax2.plot(merged["dt"], merged["actual_next_price"],
                 color="black", linewidth=2, marker="o", markersize=4,
                 label="Actual Next Price (P_t+1)")
        ax2.plot(merged["dt"], merged["predicted_next_price"],
                 color="royalblue", linewidth=2, linestyle="--", marker="s", markersize=4,
                 label="Predicted Next Price")
        ax2.plot(merged["dt"], merged["Close"],
                 color="gray", linewidth=1, linestyle=":", alpha=0.7,
                 label="Current Price (P_t)")
        ax2.set_ylabel("Price (₹)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    save_path = PLOT_DIR / f"hybrid_live_{date_range_tag}_{code}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 2: Combined return grid
# ---------------------------------------------------------------------------

def _plot_combined_grid(
    top_n: pd.DataFrame,
    df_results: pd.DataFrame,
    date_range_tag: str,
) -> None:
    if top_n.empty:
        return
    n_rows = len(top_n)
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4.5 * n_rows), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, (_, row) in zip(axes, top_n.iterrows()):
        code = row["code"]
        sd = df_results[df_results["code"] == code].sort_values("dt")
        ax.plot(sd["dt"], sd["actual_return"] * 100,
                color="black", linewidth=2, marker="o", markersize=4, label="Actual")
        ax.plot(sd["dt"], sd["predicted_return"] * 100,
                color="royalblue", linewidth=2, linestyle="--", marker="s", markersize=4,
                label="Predicted")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(
            f"{code}  |  Spearman IC: {row['spearman_ic']:.4f}  "
            f"|  Sign Acc: {row['sign_acc']:.2%}  |  MSE: {row['mse']:.6f}"
        )
        ax.set_ylabel("Return (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    fig.suptitle(
        f"Top {n_rows} Stocks by Spearman IC  |  Hybrid Model  [{date_range_tag}]",
        fontsize=13, y=0.995,
    )
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    save_path = PLOT_DIR / f"hybrid_live_top{n_rows}_{date_range_tag}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined grid: {save_path}")


# ---------------------------------------------------------------------------
# Plot 3: Daily cross-sectional rankings
# ---------------------------------------------------------------------------

def _plot_daily_rankings(df_results: pd.DataFrame, date_range_tag: str) -> None:
    dates = sorted(df_results["dt"].unique())
    n_days = len(dates)
    if n_days == 0:
        return

    fig, axes = plt.subplots(n_days, 1, figsize=(16, 3.5 * n_days), squeeze=False)
    axes = axes[:, 0]

    for ax, dt in zip(axes, dates):
        day = df_results[df_results["dt"] == dt].sort_values("predicted_return", ascending=False).copy()
        colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in day["predicted_return"]]
        x = np.arange(len(day))
        ax.bar(x, day["predicted_return"] * 100, color=colors, alpha=0.85, label="Predicted Return")
        ax.scatter(x, day["actual_return"] * 100,
                   color="black", zorder=5, s=35, marker="o", label="Actual Return")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(day["code"].str.replace(".NS", "", regex=False),
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Return (%)")
        ax.set_title(f"{pd.Timestamp(dt).strftime('%A %b %d, %Y')} — Cross-Sectional Rankings")
        ax.grid(True, alpha=0.3, axis="y")
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Daily Cross-Sectional Rankings  |  Hybrid Model  [{date_range_tag}]",
                 fontsize=13, y=1.0)
    plt.tight_layout()

    save_path = PLOT_DIR / f"hybrid_live_rankings_{date_range_tag}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved rankings chart: {save_path}")


# ---------------------------------------------------------------------------
# Plot 4: Long-short portfolio simulation
# ---------------------------------------------------------------------------

def _plot_long_short_portfolio(
    df_results: pd.DataFrame,
    n: int,
    date_range_tag: str,
) -> None:
    daily_rows = []
    for dt, group in df_results.groupby("dt"):
        group = group.sort_values("predicted_return", ascending=False)
        longs  = group.head(n)["actual_return"]
        shorts = group.tail(n)["actual_return"]
        if longs.empty or shorts.empty:
            continue
        daily_rows.append({
            "dt":        dt,
            "long_ret":  float(longs.mean()),
            "short_ret": float(shorts.mean()),
            "ls_ret":    float(longs.mean()) - float(shorts.mean()),
        })

    if not daily_rows:
        return

    df_pnl = pd.DataFrame(daily_rows).sort_values("dt")
    df_pnl["cum_ls"]         = (1 + df_pnl["ls_ret"]).cumprod() - 1
    df_pnl["cum_long"]       = (1 + df_pnl["long_ret"]).cumprod() - 1
    df_pnl["cum_short_side"] = -(1 + df_pnl["short_ret"]).cumprod() + 1

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    ax = axes[0]
    ax.plot(df_pnl["dt"], df_pnl["cum_ls"] * 100,
            color="navy", linewidth=2.5, label=f"Long-Short (top/bottom {n})")
    ax.plot(df_pnl["dt"], df_pnl["cum_long"] * 100,
            color="#2ecc71", linewidth=1.8, linestyle="--", label=f"Long only (top {n})")
    ax.plot(df_pnl["dt"], df_pnl["cum_short_side"] * 100,
            color="#e74c3c", linewidth=1.8, linestyle="--", label=f"Short side (bottom {n})")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.fill_between(df_pnl["dt"], df_pnl["cum_ls"] * 100, 0,
                    where=(df_pnl["cum_ls"] >= 0), alpha=0.08, color="green")
    ax.fill_between(df_pnl["dt"], df_pnl["cum_ls"] * 100, 0,
                    where=(df_pnl["cum_ls"] < 0), alpha=0.08, color="red")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(f"Simulated Long-Short Portfolio  |  Hybrid Model  [{date_range_tag}]",
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2 = axes[1]
    bar_colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in df_pnl["ls_ret"]]
    ax2.bar(df_pnl["dt"], df_pnl["ls_ret"] * 100, color=bar_colors, alpha=0.85, width=0.6)
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_ylabel("Daily L-S Return (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3, axis="y")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    total_ls  = df_pnl["cum_ls"].iloc[-1] * 100
    hit_rate  = (df_pnl["ls_ret"] > 0).mean() * 100
    sharpe    = (df_pnl["ls_ret"].mean() / df_pnl["ls_ret"].std() * np.sqrt(252)
                 if df_pnl["ls_ret"].std() > 0 else float("nan"))
    print(f"  L-S cumulative return : {total_ls:+.2f}%")
    print(f"  Daily hit rate        : {hit_rate:.1f}%")
    print(f"  Annualised Sharpe     : {sharpe:.2f}")

    save_path = PLOT_DIR / f"hybrid_live_longshort_{date_range_tag}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved portfolio chart: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    date_range_tag = f"{args.start_date}_to_{args.end_date}"

    data_files = sorted([p.name for p in TRAIN_DATA_DIR.glob("*.pkl")])
    if not data_files:
        raise RuntimeError(
            f"No .pkl files found in {TRAIN_DATA_DIR}. Run the THGNN data pipeline first."
        )

    file_dates = [pd.to_datetime(Path(name).stem).normalize() for name in data_files]
    start_idx = _index_for_date(file_dates, args.start_date)
    end_idx   = min(_index_for_date(file_dates, args.end_date) + 1, len(data_files))

    if end_idx <= start_idx:
        raise ValueError(
            f"No pkl files found between {args.start_date} and {args.end_date}."
        )

    actual_start = file_dates[start_idx].date()
    actual_end   = file_dates[end_idx - 1].date()
    print(f"Date range  : {actual_start} → {actual_end}  ({end_idx - start_idx} trading day(s))")

    # Load checkpoint and rebuild model
    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    print(f"Checkpoint  : {checkpoint_path}")
    checkpoint  = _load_checkpoint(checkpoint_path, device)
    state_dict  = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    cfg         = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    trained_until = checkpoint.get("split", {}).get("pre_data", "unknown")
    print(f"Model trained until : {trained_until}")

    model = _build_model(cfg, state_dict, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters          : {n_params:,}")

    dataset = AllGraphDataSampler(
        str(TRAIN_DATA_DIR), mode="val",
        data_start=0, data_middle=start_idx, data_end=end_idx,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty for the selected date range.")

    # Collect predictions
    df_results = _collect_predictions(dataset, model, device)
    if df_results.empty:
        raise RuntimeError(
            "No predictions generated. Check that pkl and daily_stock files exist for the range."
        )

    # Per-stock metrics
    metrics = []
    for code, group in df_results.groupby("code"):
        if len(group) < args.min_samples:
            continue
        pred_ret = group["predicted_return"].astype(float)
        act_ret  = group["actual_return"].astype(float)
        metrics.append({
            "code":        code,
            "spearman_ic": _spearman_ic(pred_ret, act_ret),
            "sign_acc":    float(np.mean(np.sign(pred_ret) == np.sign(act_ret))),
            "mse":         float(np.mean((act_ret - pred_ret) ** 2)),
            "n_samples":   len(group),
        })

    df_metrics = pd.DataFrame(metrics)
    if df_metrics.empty:
        raise RuntimeError("No stock metrics computed.")

    top_n = (
        df_metrics[df_metrics["spearman_ic"].notna()]
        .sort_values(by=["spearman_ic", "sign_acc", "mse"], ascending=[False, False, True])
        .head(args.top_n)
    )

    print(f"\nTop {len(top_n)} stocks by Spearman IC:")
    print(top_n[["code", "spearman_ic", "sign_acc", "mse", "n_samples"]].to_string(index=False))

    print(f"\n[1/4] Individual stock plots (return + price)...")
    for _, row in top_n.iterrows():
        code = row["code"]
        stock_data = df_results[df_results["code"] == code].sort_values("dt")
        _plot_stock(code, stock_data, row, _load_prices(code), date_range_tag)

    print(f"\n[2/4] Combined return grid...")
    _plot_combined_grid(top_n, df_results, date_range_tag)

    print(f"\n[3/4] Daily cross-sectional rankings...")
    _plot_daily_rankings(df_results, date_range_tag)

    print(f"\n[4/4] Long-short portfolio simulation...")
    _plot_long_short_portfolio(df_results, args.long_short_n, date_range_tag)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
