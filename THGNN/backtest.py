"""
backtest.py — Long-short backtest with clean financial metrics for THGNN.

Loads the best checkpoint, runs inference over a test date range, simulates
an equal-weight long-top-K / short-bottom-K portfolio, and saves metrics +
equity curve to data/backtest_results/.

Usage
-----
    # Default: top 5 long / bottom 5 short, test 2025 onwards
    python backtest.py

    # Custom
    python backtest.py --start-date 2024-01-01 --end-date 2026-02-28 --top-k 10

    # Specific checkpoint
    python backtest.py --checkpoint data/model_saved/2024-12-31_icrank_best.dat
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

from data_loader import AllGraphDataSampler
from model.Thgnn import StockHeteGAT
from trainer.trainer import extract_data


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "data_train_predict"
MODEL_DIR = DATA_DIR / "model_saved"
RESULTS_DIR = DATA_DIR / "backtest_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="THGNN backtest with financial metrics.")
    parser.add_argument("--start-date", type=str, default="2025-01-01")
    parser.add_argument("--end-date",   type=str, default="2026-12-31")
    parser.add_argument("--top-k",      type=int, default=5,
                        help="Stocks on each side of the long-short portfolio.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .dat checkpoint. Defaults to newest *_icrank_best.dat.")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--in-features", type=int, default=12)
    return parser.parse_args()


def _resolve_checkpoint(checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        p = Path(checkpoint_arg)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    candidates = sorted(MODEL_DIR.glob("*_icrank_best.dat"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No *_icrank_best.dat found in {MODEL_DIR}")
    return candidates[-1]


def _index_for_date(file_dates: list[pd.Timestamp], date_str: str, mode: str = "start") -> int:
    dt = pd.to_datetime(date_str).normalize()
    if mode == "start":
        for i, d in enumerate(file_dates):
            if d >= dt:
                return i
        return len(file_dates) - 1
    else:  # end: exclusive
        for i, d in enumerate(file_dates):
            if d > dt:
                return i
        return len(file_dates)


def _load_model(ckpt_path: Path, device: torch.device, in_features: int) -> StockHeteGAT:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model = StockHeteGAT(
        in_features=int(cfg.get("in_features", in_features)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_heads=int(cfg.get("num_heads", 4)),
        num_layers=int(cfg.get("num_layers", 1)),
        out_features=int(cfg.get("out_features", 32)),
        predictor_out_dim=state_dict["predictor.0.weight"].shape[0],
        predictor_activation=None,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path.name}  (epoch {ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'})")
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _sharpe(returns: np.ndarray, periods: int = 252) -> float:
    if returns.std() == 0:
        return float("nan")
    return float(returns.mean() / returns.std() * np.sqrt(periods))


def _max_drawdown(cum_returns: np.ndarray) -> float:
    """Max drawdown from peak equity (as a negative fraction)."""
    equity = (1 + cum_returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def _calmar(ann_return: float, max_dd: float) -> float:
    if max_dd == 0:
        return float("nan")
    return ann_return / abs(max_dd)


def _ic_stats(preds: np.ndarray, actuals: np.ndarray) -> dict:
    """Spearman IC and t-stat across all cross-sections."""
    ics = []
    for p, a in zip(preds, actuals):
        if len(p) < 5:
            continue
        r, _ = stats.spearmanr(p, a)
        if not np.isnan(r):
            ics.append(r)
    ics = np.array(ics)
    if len(ics) == 0:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "ic_ir": float("nan"),
                "ic_tstat": float("nan"), "ic_hit_rate": float("nan")}
    tstat = float(np.mean(ics) / (np.std(ics) / np.sqrt(len(ics)))) if np.std(ics) > 0 else float("nan")
    return {
        "ic_mean":     float(np.mean(ics)),
        "ic_std":      float(np.std(ics)),
        "ic_ir":       float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else float("nan"),
        "ic_tstat":    tstat,
        "ic_hit_rate": float((ics > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = _resolve_checkpoint(args.checkpoint)
    model = _load_model(ckpt_path, device, args.in_features)

    # Resolve file indices
    data_files = sorted([p.name for p in TRAIN_DATA_DIR.glob("*.pkl")])
    if not data_files:
        raise RuntimeError(f"No .pkl files found in {TRAIN_DATA_DIR}")
    file_dates = [pd.to_datetime(Path(n).stem).normalize() for n in data_files]
    start_idx = _index_for_date(file_dates, args.start_date, "start")
    end_idx   = _index_for_date(file_dates, args.end_date,   "end")
    end_idx   = min(end_idx, len(data_files))

    if end_idx <= start_idx:
        raise ValueError(f"No data in [{args.start_date}, {args.end_date}]")

    print(f"Backtesting {end_idx - start_idx} trading days: "
          f"{file_dates[start_idx].date()} → {file_dates[end_idx - 1].date()}")

    ds = AllGraphDataSampler(
        str(TRAIN_DATA_DIR), mode="val",
        data_start=0, data_middle=start_idx, data_end=end_idx,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, pin_memory=False, collate_fn=lambda x: x)

    # Run inference
    daily_preds, daily_actuals, daily_dates = [], [], []
    for i, batch in enumerate(tqdm(loader, desc="Inference")):
        data = batch[0]
        try:
            pos_adj, neg_adj, features, labels, mask = extract_data(data, device)
        except Exception as e:
            print(f"  Skip sample {i}: {e}")
            continue
        with torch.no_grad():
            preds = model(features, pos_adj, neg_adj).squeeze(-1)

        mask_t = torch.tensor(mask, device=device, dtype=torch.bool) if not isinstance(mask, torch.Tensor) else mask.to(device)
        p = preds[mask_t].cpu().numpy()
        a = labels[mask_t].cpu().numpy() if labels.dim() == 1 else labels[mask_t, 0].cpu().numpy()

        if len(p) >= args.top_k * 2:
            daily_preds.append(p)
            daily_actuals.append(a)
            daily_dates.append(file_dates[start_idx + i])

    if not daily_preds:
        print("No valid predictions. Check data and checkpoint compatibility.")
        return

    # Portfolio simulation: long top-K, short bottom-K by predicted return
    k = args.top_k
    daily_ls_returns = []
    daily_long_returns = []
    daily_short_returns = []

    for pred, actual in zip(daily_preds, daily_actuals):
        order = np.argsort(pred)[::-1]  # descending: highest predicted return first
        long_idx  = order[:k]
        short_idx = order[-k:]
        long_ret  = actual[long_idx].mean()
        short_ret = actual[short_idx].mean()
        ls_ret    = long_ret - short_ret
        daily_ls_returns.append(ls_ret)
        daily_long_returns.append(long_ret)
        daily_short_returns.append(short_ret)

    ls_ret   = np.array(daily_ls_returns)
    long_ret = np.array(daily_long_returns)
    short_ret = np.array(daily_short_returns)

    # Cumulative returns
    cum_ls    = np.cumprod(1 + ls_ret) - 1
    cum_long  = np.cumprod(1 + long_ret) - 1

    # Financial metrics
    n_days = len(ls_ret)
    ann_factor = 252 / n_days
    total_return_ls  = float(cum_ls[-1])
    ann_return_ls    = float((1 + total_return_ls) ** ann_factor - 1)
    volatility_ls    = float(ls_ret.std() * np.sqrt(252))
    sharpe_ls        = _sharpe(ls_ret)
    sortino_denom    = ls_ret[ls_ret < 0].std() if (ls_ret < 0).any() else 1e-9
    sortino_ls       = float(ls_ret.mean() / sortino_denom * np.sqrt(252))
    max_dd_ls        = _max_drawdown(ls_ret)
    calmar_ls        = _calmar(ann_return_ls, max_dd_ls)
    win_rate_ls      = float((ls_ret > 0).mean())

    # IC stats
    ic_stats = _ic_stats(daily_preds, daily_actuals)

    # Long-only metrics
    total_return_long = float(cum_long[-1])
    ann_return_long   = float((1 + total_return_long) ** ann_factor - 1)
    sharpe_long       = _sharpe(long_ret)
    win_rate_long     = float((long_ret > 0).mean())

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"backtest_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics report
    sep = "=" * 68
    report_lines = [
        sep,
        "THGNN BACKTEST PERFORMANCE REPORT",
        sep,
        f"",
        f"Generated on:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Checkpoint:      {ckpt_path.name}",
        f"Period:          {daily_dates[0].date()} → {daily_dates[-1].date()}  ({n_days} days)",
        f"Universe:        top-{k} long / bottom-{k} short (equal weight)",
        f"",
        "LONG-SHORT PORTFOLIO",
        "-" * 68,
        f"Total Return (%):            {total_return_ls * 100:>12.4f}",
        f"Annualized Return (%):       {ann_return_ls   * 100:>12.4f}",
        f"Volatility (% ann):          {volatility_ls   * 100:>12.4f}",
        f"Sharpe Ratio (ann):          {sharpe_ls             :>12.4f}",
        f"Sortino Ratio (ann):         {sortino_ls            :>12.4f}",
        f"Max Drawdown (%):            {max_dd_ls       * 100:>12.4f}",
        f"Calmar Ratio:                {calmar_ls             :>12.4f}",
        f"Daily Win Rate (%):          {win_rate_ls     * 100:>12.4f}",
        f"",
        "LONG-ONLY (top-K) PORTFOLIO",
        "-" * 68,
        f"Total Return (%):            {total_return_long * 100:>12.4f}",
        f"Annualized Return (%):       {ann_return_long   * 100:>12.4f}",
        f"Sharpe Ratio (ann):          {sharpe_long             :>12.4f}",
        f"Daily Win Rate (%):          {win_rate_long     * 100:>12.4f}",
        f"",
        "INFORMATION COEFFICIENT",
        "-" * 68,
        f"Mean IC:                     {ic_stats['ic_mean']:>12.4f}",
        f"IC Std:                      {ic_stats['ic_std']:>12.4f}",
        f"IC IR (ICIR):                {ic_stats['ic_ir']:>12.4f}",
        f"IC t-stat:                   {ic_stats['ic_tstat']:>12.4f}",
        f"IC Hit Rate (%):             {ic_stats['ic_hit_rate'] * 100:>12.4f}",
        sep,
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    (out_dir / "metrics_report.txt").write_text(report_text)

    # JSON for programmatic access
    metrics_json = {
        "checkpoint": str(ckpt_path),
        "start_date": str(daily_dates[0].date()),
        "end_date": str(daily_dates[-1].date()),
        "n_days": n_days,
        "top_k": k,
        "longshort": {
            "total_return_pct": round(total_return_ls * 100, 4),
            "ann_return_pct": round(ann_return_ls * 100, 4),
            "volatility_ann_pct": round(volatility_ls * 100, 4),
            "sharpe": round(sharpe_ls, 4),
            "sortino": round(sortino_ls, 4),
            "max_drawdown_pct": round(max_dd_ls * 100, 4),
            "calmar": round(calmar_ls, 4),
            "win_rate_pct": round(win_rate_ls * 100, 4),
        },
        "long_only": {
            "total_return_pct": round(total_return_long * 100, 4),
            "ann_return_pct": round(ann_return_long * 100, 4),
            "sharpe": round(sharpe_long, 4),
            "win_rate_pct": round(win_rate_long * 100, 4),
        },
        "ic": {k: round(v, 4) if not np.isnan(v) else None for k, v in ic_stats.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2))

    # Equity curve plot
    dates = daily_dates
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    ax = axes[0]
    ax.plot(dates, cum_ls * 100, color="navy", lw=2, label=f"Long-Short (top/bottom {k})")
    ax.plot(dates, cum_long * 100, color="#2ecc71", lw=1.5, ls="--", label=f"Long only (top {k})")
    ax.fill_between(dates, cum_ls * 100, 0, where=np.array(cum_ls) >= 0, alpha=0.1, color="green")
    ax.fill_between(dates, cum_ls * 100, 0, where=np.array(cum_ls) < 0, alpha=0.1, color="red")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(f"THGNN Backtest: {daily_dates[0].date()} → {daily_dates[-1].date()}", fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    bar_colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in ls_ret]
    ax.bar(dates, ls_ret * 100, color=bar_colors, alpha=0.8, width=0.6)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Daily L-S Return (%)")
    ax.grid(True, alpha=0.3, axis="y")

    rolling_ic = pd.Series(
        [stats.spearmanr(p, a)[0] for p, a in zip(daily_preds, daily_actuals)],
        index=dates
    ).rolling(20, min_periods=5).mean()
    ax = axes[2]
    ax.plot(dates, rolling_ic, color="purple", lw=1.5, label="20-day rolling IC")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.axhline(ic_stats["ic_mean"], color="purple", lw=1, ls="--", alpha=0.5,
               label=f"Mean IC={ic_stats['ic_mean']:.3f}")
    ax.set_ylabel("Spearman IC (rolling 20d)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "equity_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nResults saved to: {out_dir}")
    print(f"  metrics_report.txt, metrics.json, equity_curve.png")


if __name__ == "__main__":
    main()
