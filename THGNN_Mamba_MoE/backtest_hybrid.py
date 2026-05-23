"""
backtest_hybrid.py - Long-short backtest for the Hybrid THGNN x MaGNet Mamba+MoE model.

Loads the best hybrid checkpoint, runs inference over a date range, simulates
an equal-weight long-top-K / short-bottom-K portfolio, and saves metrics +
equity curve to THGNN/data/backtest_results/.

Usage
-----
    # Default: 2024-01-01 → 2026-02-28, top-5 L/S
    python backtest_hybrid.py

    # Custom period and portfolio size
    python backtest_hybrid.py --start-date 2024-06-01 --end-date 2026-02-28 --top-k 10

    # Specific checkpoint
    python backtest_hybrid.py --checkpoint ../THGNN/data/model_saved/2023-12-29_mamba_moe_best.dat

    # GNN-only (skip news), save results to a custom directory
    python backtest_hybrid.py --out-dir /tmp/my_backtest
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

def _patch_pathlib() -> None:
    import types, pathlib as _pl
    if "pathlib._local" not in sys.modules:
        shim = types.ModuleType("pathlib._local")
        shim.PosixPath   = _pl.PurePosixPath   # type: ignore[attr-defined]
        shim.WindowsPath = _pl.PureWindowsPath  # type: ignore[attr-defined]
        sys.modules["pathlib._local"] = shim
    if hasattr(_pl, "PosixPath"):
        _pl.PosixPath = _pl.PurePosixPath      # type: ignore[attr-defined]
_patch_pathlib()

from data_loader import AllGraphDataSampler
from model.hybrid_model import HybridStockModel


_HERE          = Path(__file__).resolve().parent
_THGNN_DIR     = _HERE.parent / "THGNN"
DATA_DIR        = _THGNN_DIR / "data"
TRAIN_DATA_DIR  = DATA_DIR / "data_train_predict"
DAILY_STOCK_DIR = DATA_DIR / "daily_stock"
MODEL_DIR       = DATA_DIR / "model_saved"
RESULTS_DIR     = DATA_DIR / "backtest_results"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest the Hybrid THGNN x MaGNet Mamba+MoE model with financial metrics."
    )
    p.add_argument("--start-date",  type=str, default="2024-01-01",
                   help="First trading day to include in the backtest.")
    p.add_argument("--end-date",    type=str, default="2026-02-28",
                   help="Last trading day to include (inclusive).")
    p.add_argument("--top-k",       type=int, default=5,
                   help="Stocks on each side of the long-short portfolio.")
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Path to a *_mamba_moe_best.dat file. "
                        "Auto-selects newest if not given.")
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--device",      type=str, default=None,
                   help="'cuda' or 'cpu'. Auto-detected if omitted.")
    p.add_argument("--out-dir",     type=str, default=None,
                   help="Override output directory (default: THGNN/data/backtest_results/<ts>).")
    p.add_argument("--cost-bps",    type=float, nargs="+", default=[0, 5, 10, 20],
                   help="Transaction cost levels in basis points for sensitivity table.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_checkpoint(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    candidates = sorted(MODEL_DIR.glob("*_mamba_moe_best.dat"),
                        key=lambda x: x.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No *_mamba_moe_best.dat found in {MODEL_DIR}")
    return candidates[-1]


def _index_for_date(file_dates: list[pd.Timestamp], date_str: str,
                    mode: str = "start") -> int:
    dt = pd.to_datetime(date_str).normalize()
    if mode == "start":
        for i, d in enumerate(file_dates):
            if d >= dt:
                return i
        return len(file_dates) - 1
    else:  # end: exclusive upper bound
        for i, d in enumerate(file_dates):
            if d > dt:
                return i
        return len(file_dates)


def _load_model(ckpt_path: Path, device: torch.device) -> HybridStockModel:
    import pathlib
    # Checkpoints saved on Linux contain PosixPath objects; remap for Windows loading
    _orig = getattr(pathlib, "PosixPath", None)
    try:
        if _orig is not None:
            pathlib.PosixPath = pathlib.PurePosixPath  # type: ignore[attr-defined]
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    finally:
        if _orig is not None:
            pathlib.PosixPath = _orig  # type: ignore[attr-defined]
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg   = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    def _i(key: str, default: int) -> int:
        return int(cfg.get(key, default))
    def _f(key: str, default: float) -> float:
        return float(cfg.get(key, default))

    model = HybridStockModel(
        in_features          = _i("in_features",          12),
        embed_dim            = _i("embed_dim",             64),
        num_mage_layers      = _i("num_mage_layers",        1),
        num_moe_experts      = _i("num_moe_experts",        4),
        num_mha_heads        = _i("num_mha_heads",          2),
        gat_heads            = _i("gat_heads",              8),
        gat_out_features     = _i("gat_out_features",       8),
        num_hyper_edges      = _i("num_hyper_edges",       32),
        num_tch_hyper_edges  = _i("num_tch_hyper_edges",   32),
        num_tch_heads        = _i("num_tch_heads",          4),
        dropout              = 0.0,   # no dropout at inference
        predictor_out_dim    = 1,
        predictor_activation = None,
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    ic    = ckpt.get("test_ic", float("nan")) if isinstance(ckpt, dict) else float("nan")
    print(f"Loaded: {ckpt_path.name}  (epoch {epoch}, best IC={ic:.4f})")
    return model


def _extract(data_dict: dict, device: torch.device):
    """Move a pkl sample to device, strip DataLoader batch dims."""
    pos_adj  = data_dict["pos_adj"].to(device)
    neg_adj  = data_dict["neg_adj"].to(device)
    features = data_dict["features"].to(device)
    labels   = data_dict["labels"].to(device)

    def _sq(t: torch.Tensor, min_dims: int) -> torch.Tensor:
        while t.dim() > min_dims and t.size(0) == 1:
            t = t.squeeze(0)
        return t

    features = _sq(features, 3)
    pos_adj  = _sq(pos_adj,  2)
    neg_adj  = _sq(neg_adj,  2)
    if labels.dim() > 1 and labels.size(-1) == 1:
        labels = labels.squeeze(-1)

    mask = data_dict["mask"]
    if isinstance(mask, torch.Tensor):
        mask = mask.to(device).bool()
    else:
        mask = torch.tensor(mask, device=device, dtype=torch.bool)
    return pos_adj, neg_adj, features, labels, mask


def _load_tickers(date: pd.Timestamp) -> list[str] | None:
    """Return ordered ticker list for `date` from daily_stock CSV, or None if missing."""
    csv_path = DAILY_STOCK_DIR / f"{date.date()}.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if "code" in df.columns:
            return df["code"].tolist()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------

def _sharpe(returns: np.ndarray, periods: int = 252) -> float:
    std = returns.std()
    return float(returns.mean() / std * np.sqrt(periods)) if std > 0 else float("nan")


def _sortino(returns: np.ndarray, periods: int = 252) -> float:
    downside = returns[returns < 0]
    dstd = downside.std() if len(downside) > 0 else 1e-9
    return float(returns.mean() / dstd * np.sqrt(periods)) if dstd > 0 else float("nan")


def _max_drawdown(daily_returns: np.ndarray) -> float:
    equity = np.cumprod(1 + daily_returns)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    return float(dd.min())


def _calmar(ann_return: float, max_dd: float) -> float:
    return ann_return / abs(max_dd) if max_dd != 0 else float("nan")


def _ic_series(preds: list[np.ndarray], actuals: list[np.ndarray]) -> np.ndarray:
    ics = []
    for p, a in zip(preds, actuals):
        if len(p) < 5:
            ics.append(float("nan"))
            continue
        r, _ = stats.spearmanr(p, a)
        ics.append(r)
    return np.array(ics, dtype=float)


def _ic_stats(ic_arr: np.ndarray) -> dict:
    valid = ic_arr[~np.isnan(ic_arr)]
    if len(valid) == 0:
        return {k: float("nan") for k in
                ("ic_mean", "ic_std", "ic_ir", "ic_tstat", "ic_hit_rate")}
    tstat = (float(np.mean(valid) / (np.std(valid) / np.sqrt(len(valid))))
             if np.std(valid) > 0 else float("nan"))
    return {
        "ic_mean":     float(np.mean(valid)),
        "ic_std":      float(np.std(valid)),
        "ic_ir":       float(np.mean(valid) / np.std(valid)) if np.std(valid) > 0 else float("nan"),
        "ic_tstat":    tstat,
        "ic_hit_rate": float((valid > 0).mean()),
        "n_days":      int(len(valid)),
    }


def _quintile_returns(preds: list[np.ndarray],
                      actuals: list[np.ndarray]) -> dict[str, list[float]]:
    """Average actual return per predicted-rank quintile (Q1=top, Q5=bottom)."""
    q_returns: dict[str, list[float]] = {f"Q{i+1}": [] for i in range(5)}
    for p, a in zip(preds, actuals):
        n = len(p)
        if n < 10:
            continue
        order = np.argsort(p)[::-1]  # descending: best predicted first
        for q in range(5):
            lo = q * n // 5
            hi = (q + 1) * n // 5
            q_returns[f"Q{q+1}"].append(float(a[order[lo:hi]].mean()))
    return q_returns


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_equity_curve(
    dates: list[pd.Timestamp],
    ls_ret: np.ndarray,
    long_ret: np.ndarray,
    short_ret: np.ndarray,
    ic_series: np.ndarray,
    ic_mean: float,
    top_k: int,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    cum_ls    = np.cumprod(1 + ls_ret)    - 1
    cum_long  = np.cumprod(1 + long_ret)  - 1
    cum_short = np.cumprod(1 + short_ret) - 1

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(
        f"Hybrid THGNN x MaGNet Mamba+MoE Backtest{title_suffix}\n"
        f"{dates[0].date()} → {dates[-1].date()}",
        fontsize=13,
    )

    # Panel 1: cumulative returns
    ax = axes[0]
    ax.plot(dates, cum_ls   * 100, color="navy",     lw=2,   label=f"L-S (top/bot {top_k})")
    ax.plot(dates, cum_long * 100, color="#2ecc71",  lw=1.5, ls="--", label=f"Long only (top {top_k})")
    ax.fill_between(dates, cum_ls * 100, 0,
                    where=(np.array(cum_ls) >= 0), alpha=0.1, color="green")
    ax.fill_between(dates, cum_ls * 100, 0,
                    where=(np.array(cum_ls) <  0), alpha=0.1, color="red")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 2: daily L-S bar chart
    ax = axes[1]
    colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in ls_ret]
    ax.bar(dates, ls_ret * 100, color=colors, alpha=0.75, width=0.6)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Daily L-S Return (%)")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: rolling IC
    ic_s = pd.Series(ic_series, index=dates)
    rolling_ic = ic_s.rolling(60, min_periods=10).mean()
    ax = axes[2]
    ax.plot(dates, ic_series,  color="thistle",  lw=0.8, alpha=0.6, label="Daily IC")
    ax.plot(dates, rolling_ic, color="purple",   lw=1.5, label="60-day rolling IC")
    ax.axhline(0,       color="gray",   lw=0.8, ls=":")
    ax.axhline(ic_mean, color="purple", lw=1.0, ls="--", alpha=0.5,
               label=f"Mean IC = {ic_mean:.3f}")
    ax.set_ylabel("Spearman IC")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_quintiles(
    q_returns: dict[str, list[float]],
    out_path: Path,
) -> None:
    means = [np.mean(v) * 252 * 100 for v in q_returns.values()]  # annualised %
    labels = list(q_returns.keys())
    colors = ["#27ae60", "#2ecc71", "#f39c12", "#e74c3c", "#c0392b"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, color=colors, alpha=0.85)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Predicted-Rank Quintile  (Q1 = highest predicted return)")
    ax.set_ylabel("Annualised Return (%)")
    ax.set_title("Quintile Return Analysis - Hybrid THGNN x MaGNet Mamba+MoE")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.5 if val >= 0 else -1.5),
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ic_histogram(ic_arr: np.ndarray, ic_mean: float, out_path: Path) -> None:
    valid = ic_arr[~np.isnan(ic_arr)]
    from scipy.stats import gaussian_kde
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid, bins=40, color="steelblue", alpha=0.7, edgecolor="white", density=True)
    kde = gaussian_kde(valid)
    xs  = np.linspace(valid.min() - 0.02, valid.max() + 0.02, 300)
    ax.plot(xs, kde(xs), color="navy", lw=2, label="KDE")
    ax.axvline(ic_mean, color="red",  lw=1.5, ls="--", label=f"Mean IC = {ic_mean:.4f}")
    ax.axvline(0,       color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Daily Spearman IC")
    ax.set_ylabel("Density")
    ax.set_title(f"IC Distribution — Mamba--MoE  (n={len(valid)} days)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cost_sensitivity(
    cost_table: list[dict],
    out_path: Path,
) -> None:
    bps_vals    = [r["cost_bps"]    for r in cost_table]
    sharpe_vals = [r["net_sharpe"]  for r in cost_table]
    total_vals  = [r["net_total_pct"] for r in cost_table]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Transaction Cost Sensitivity - Hybrid THGNN x MaGNet Mamba+MoE", fontsize=12)

    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in sharpe_vals]
    bars1 = ax1.bar([str(b) for b in bps_vals], sharpe_vals, color=colors, alpha=0.85)
    ax1.axhline(0, color="gray", lw=0.8, ls=":")
    ax1.set_xlabel("Transaction Cost (bps one-way)")
    ax1.set_ylabel("Net Sharpe Ratio (ann)")
    ax1.set_title("Net Sharpe vs Cost")
    for bar, val in zip(bars1, sharpe_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 val + (0.03 if val >= 0 else -0.06),
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    colors2 = ["#2ecc71" if t > 0 else "#e74c3c" for t in total_vals]
    bars2 = ax2.bar([str(b) for b in bps_vals], total_vals, color=colors2, alpha=0.85)
    ax2.axhline(0, color="gray", lw=0.8, ls=":")
    ax2.set_xlabel("Transaction Cost (bps one-way)")
    ax2.set_ylabel("Net Total Return (%)")
    ax2.set_title("Net Total Return vs Cost")
    for bar, val in zip(bars2, total_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + (0.5 if val >= 0 else -1.5),
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    ckpt_path = _resolve_checkpoint(args.checkpoint)
    model     = _load_model(ckpt_path, device)

    # Resolve file index range
    data_files = sorted([p.name for p in TRAIN_DATA_DIR.glob("*.pkl")])
    if not data_files:
        raise RuntimeError(f"No .pkl files found in {TRAIN_DATA_DIR}")
    file_dates = [pd.to_datetime(Path(n).stem).normalize() for n in data_files]

    start_idx = _index_for_date(file_dates, args.start_date, "start")
    end_idx   = _index_for_date(file_dates, args.end_date,   "end")
    end_idx   = min(end_idx, len(data_files))

    if end_idx <= start_idx:
        raise ValueError(
            f"No data found in [{args.start_date}, {args.end_date}]. "
            f"Data spans {file_dates[0].date()} → {file_dates[-1].date()}."
        )
    print(
        f"Backtesting {end_idx - start_idx} trading days: "
        f"{file_dates[start_idx].date()} → {file_dates[end_idx - 1].date()}"
    )

    ds = AllGraphDataSampler(
        str(TRAIN_DATA_DIR),
        mode="val",
        data_start=0,
        data_middle=start_idx,
        data_end=end_idx,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, pin_memory=False, collate_fn=lambda x: x
    )

    # ---- Inference ----
    daily_preds:   list[np.ndarray]    = []
    daily_actuals: list[np.ndarray]    = []
    daily_dates:   list[pd.Timestamp]  = []
    daily_tickers: list[list[str] | None] = []   # ticker names per day for accurate turnover

    for i, batch in enumerate(tqdm(loader, desc="Inference")):
        data = batch[0]
        try:
            pos_adj, neg_adj, features, labels, mask = _extract(data, device)
        except Exception as e:
            print(f"  Skip sample {i}: {e}")
            continue

        with torch.no_grad():
            logits = model(features, pos_adj, neg_adj).squeeze(-1)   # (N,)

        mask_np = mask.cpu().numpy().astype(bool)
        p = logits[mask].cpu().numpy()
        a = labels[mask].cpu().numpy()
        if a.ndim > 1:
            a = a[:, 0]   # take horizon-0 label if multi-horizon

        if len(p) < args.top_k * 2:
            continue

        day_date = file_dates[start_idx + i]
        tickers_full = _load_tickers(day_date)
        # Align tickers to the masked subset, if available and length matches
        if tickers_full is not None and len(tickers_full) == len(mask_np):
            tickers_masked = [tickers_full[j] for j in range(len(mask_np)) if mask_np[j]]
        elif tickers_full is not None and len(tickers_full) == mask_np.sum():
            tickers_masked = tickers_full  # already filtered
        else:
            tickers_masked = None   # fall back to integer indices

        daily_preds.append(p)
        daily_actuals.append(a)
        daily_dates.append(day_date)
        daily_tickers.append(tickers_masked)

    if not daily_preds:
        print("No valid prediction days. Check checkpoint/data compatibility.")
        return

    n_days = len(daily_preds)
    print(f"Valid prediction days: {n_days}")

    # ---- Portfolio simulation ----
    # Dollar-neutral assumption: 50% long + 50% short = 100% gross exposure.
    # ls_ret = 0.5 * (long_ret - short_ret) so returns are on a per-unit-equity basis.
    k = args.top_k
    ls_ret      = np.empty(n_days)
    long_ret    = np.empty(n_days)
    short_ret   = np.empty(n_days)
    turnover    = np.empty(n_days)   # fraction of portfolio replaced each day

    prev_long_ids:  set = set()
    prev_short_ids: set = set()
    has_tickers = any(t is not None for t in daily_tickers)
    if not has_tickers:
        print("  Note: daily_stock CSVs not found — turnover uses positional indices.")

    for t, (pred, actual) in enumerate(zip(daily_preds, daily_actuals)):
        order     = np.argsort(pred)[::-1]
        long_idx  = order[:k]
        short_idx = order[-k:]

        long_ret[t]  = actual[long_idx].mean()
        short_ret[t] = actual[short_idx].mean()
        # Dollar-neutral: 0.5x long + 0.5x short
        ls_ret[t]    = 0.5 * (long_ret[t] - short_ret[t])

        # Turnover: fraction of positions replaced vs previous day.
        # Use ticker names when available (immune to universe reordering);
        # fall back to positional integers otherwise.
        tickers = daily_tickers[t]
        if tickers is not None:
            cur_long  = {tickers[j] for j in long_idx.tolist()}
            cur_short = {tickers[j] for j in short_idx.tolist()}
        else:
            cur_long  = set(long_idx.tolist())
            cur_short = set(short_idx.tolist())

        if t == 0:
            turnover[t] = 1.0   # first day: full deployment
        else:
            new_long  = len(cur_long  - prev_long_ids)
            new_short = len(cur_short - prev_short_ids)
            turnover[t] = (new_long + new_short) / (2 * k)
        prev_long_ids  = cur_long
        prev_short_ids = cur_short

    # ---- Financial metrics (gross, 0 cost) ----
    ann_factor       = 252 / n_days
    cum_ls           = np.cumprod(1 + ls_ret) - 1
    total_return_ls  = float(cum_ls[-1])
    ann_return_ls    = float((1 + total_return_ls) ** ann_factor - 1)
    vol_ls           = float(ls_ret.std() * np.sqrt(252))
    sharpe_ls        = _sharpe(ls_ret)
    sortino_ls       = _sortino(ls_ret)
    max_dd_ls        = _max_drawdown(ls_ret)
    calmar_ls        = _calmar(ann_return_ls, max_dd_ls)
    win_rate_ls      = float((ls_ret > 0).mean())

    cum_long          = np.cumprod(1 + long_ret)  - 1
    total_return_long = float(cum_long[-1])
    ann_return_long   = float((1 + total_return_long) ** ann_factor - 1)
    sharpe_long       = _sharpe(long_ret)
    win_rate_long     = float((long_ret > 0).mean())

    # ---- Turnover stats ----
    avg_turnover     = float(turnover[1:].mean())   # skip day-0 full deploy
    ann_turnover     = avg_turnover * 252

    # ---- Transaction cost sensitivity ----
    # cost model: each position change costs cost_bps bps one-way.
    # daily cost = turnover[t] * cost_bps/10000 (applied to 100% gross book)
    cost_table: list[dict] = []
    for bps in args.cost_bps:
        daily_cost   = turnover * (bps / 10_000)
        net_ret      = ls_ret - daily_cost
        cum_net      = np.cumprod(1 + net_ret) - 1
        net_total    = float(cum_net[-1])
        net_ann      = float((1 + net_total) ** ann_factor - 1)
        net_sharpe   = _sharpe(net_ret)
        net_maxdd    = _max_drawdown(net_ret)
        cost_table.append({
            "cost_bps":       bps,
            "net_total_pct":  round(net_total * 100, 2),
            "net_ann_pct":    round(net_ann   * 100, 2),
            "net_sharpe":     round(net_sharpe,       4),
            "net_maxdd_pct":  round(net_maxdd * 100,  2),
        })

    # ---- IC metrics ----
    ic_arr   = _ic_series(daily_preds, daily_actuals)
    ic_stats = _ic_stats(ic_arr)

    # ---- Quintile analysis ----
    q_returns = _quintile_returns(daily_preds, daily_actuals)
    q_spread  = np.mean(q_returns["Q1"]) - np.mean(q_returns["Q5"])

    # ---- Yearly breakdown ----
    yearly: dict[int, dict] = {}
    dates_arr = pd.DatetimeIndex(daily_dates)
    for yr in sorted(set(dates_arr.year)):
        mask_yr  = dates_arr.year == yr
        r_yr     = ls_ret[mask_yr]
        n_yr     = int(mask_yr.sum())
        af_yr    = 252 / n_yr
        tot_yr   = float(np.cumprod(1 + r_yr)[-1] - 1)
        ann_yr   = float((1 + tot_yr) ** af_yr - 1)
        sh_yr    = _sharpe(r_yr)
        dd_yr    = _max_drawdown(r_yr)
        wr_yr    = float((r_yr > 0).mean())
        ic_yr_arr = np.array([
            stats.spearmanr(p, a)[0]
            for p, a in zip(
                [daily_preds[i]   for i in range(n_days) if dates_arr[i].year == yr],
                [daily_actuals[i] for i in range(n_days) if dates_arr[i].year == yr],
            ) if len(p) >= 5
        ])
        yearly[yr] = {
            "n_days": n_yr,
            "total_pct":  round(tot_yr * 100, 2),
            "ann_pct":    round(ann_yr * 100, 2),
            "sharpe":     round(sh_yr,         4),
            "max_dd_pct": round(dd_yr * 100,   2),
            "win_rate":   round(wr_yr * 100,   2),
            "mean_ic":    round(float(np.nanmean(ic_yr_arr)), 4) if len(ic_yr_arr) else float("nan"),
        }

    # ---- Output directory ----
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = RESULTS_DIR / f"hybrid_backtest_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Text report ----
    sep = "=" * 70
    report_lines = [
        sep,
        "HYBRID THGNN x MaGNet Mamba+MoE  BACKTEST REPORT",
        sep,
        "",
        f"Generated:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Checkpoint:  {ckpt_path.name}",
        f"Period:      {daily_dates[0].date()} -> {daily_dates[-1].date()}  ({n_days} days)",
        f"Portfolio:   top-{k} long / bottom-{k} short (equal weight, dollar-neutral)",
        f"Note:        Returns on equity basis (0.5x long + 0.5x short, 100% gross).",
        "",
        "LONG-SHORT PORTFOLIO  (gross, 0 transaction cost)",
        "-" * 70,
        f"Total Return (%):            {total_return_ls  * 100:>12.4f}",
        f"Annualised Return (%):       {ann_return_ls    * 100:>12.4f}",
        f"Annualised Volatility (%):   {vol_ls           * 100:>12.4f}",
        f"Sharpe Ratio (ann):          {sharpe_ls              :>12.4f}",
        f"Sortino Ratio (ann):         {sortino_ls             :>12.4f}",
        f"Max Drawdown (%):            {max_dd_ls        * 100:>12.4f}",
        f"Calmar Ratio:                {calmar_ls              :>12.4f}",
        f"Daily Win Rate (%):          {win_rate_ls      * 100:>12.4f}",
        "",
        "TURNOVER",
        "-" * 70,
        f"Avg daily turnover (%):      {avg_turnover     * 100:>12.2f}",
        f"Annualised turnover (x):     {ann_turnover          :>12.2f}",
        "",
        "TRANSACTION COST SENSITIVITY  (net of costs, dollar-neutral)",
        "-" * 70,
        f"{'Cost (bps)':<12} {'Net Total%':>11} {'Net Ann%':>10} {'Net Sharpe':>11} {'Max DD%':>9}",
        "-" * 70,
    ]
    for row in cost_table:
        report_lines.append(
            f"{row['cost_bps']:<12.0f} "
            f"{row['net_total_pct']:>11.2f} "
            f"{row['net_ann_pct']:>10.2f} "
            f"{row['net_sharpe']:>11.4f} "
            f"{row['net_maxdd_pct']:>9.2f}"
        )
    report_lines += [
        "",
        "LONG-ONLY PORTFOLIO  (top-K, gross)",
        "-" * 70,
        f"Total Return (%):            {total_return_long * 100:>12.4f}",
        f"Annualised Return (%):       {ann_return_long   * 100:>12.4f}",
        f"Sharpe Ratio (ann):          {sharpe_long             :>12.4f}",
        f"Daily Win Rate (%):          {win_rate_long     * 100:>12.4f}",
        "",
        "INFORMATION COEFFICIENT  (friction-independent)",
        "-" * 70,
        f"Mean IC:                     {ic_stats['ic_mean']      :>12.4f}",
        f"IC Std Dev:                  {ic_stats['ic_std']       :>12.4f}",
        f"ICIR  (mean/std):            {ic_stats['ic_ir']        :>12.4f}",
        f"IC t-stat:                   {ic_stats['ic_tstat']     :>12.4f}",
        f"IC Hit Rate (%):             {ic_stats['ic_hit_rate'] * 100:>12.4f}",
        "",
        "QUINTILE SPREAD  (friction-independent)",
        "-" * 70,
        f"Q1-Q5 daily spread:          {q_spread         * 100:>12.4f} %",
        f"Q1-Q5 ann. spread:           {q_spread * 252   * 100:>12.4f} %",
    ]
    for q, vals in q_returns.items():
        ann_q = np.mean(vals) * 252 * 100
        report_lines.append(f"  {q} ann. return:             {ann_q:>12.2f} %")
    report_lines += [
        "",
        "YEARLY BREAKDOWN  (gross, dollar-neutral)",
        "-" * 70,
        f"{'Year':<6} {'Days':>5} {'Total%':>8} {'Ann%':>8} {'Sharpe':>8} {'MaxDD%':>8} {'WinRate%':>9} {'MeanIC':>8}",
        "-" * 70,
    ]
    for yr, d in yearly.items():
        report_lines.append(
            f"{yr:<6} {d['n_days']:>5} {d['total_pct']:>8.2f} {d['ann_pct']:>8.2f} "
            f"{d['sharpe']:>8.4f} {d['max_dd_pct']:>8.2f} {d['win_rate']:>9.2f} {d['mean_ic']:>8.4f}"
        )
    report_lines.append(sep)

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    (out_dir / "metrics_report.txt").write_text(report_text, encoding="utf-8")

    # ---- JSON ----
    metrics_json = {
        "checkpoint":  str(ckpt_path),
        "start_date":  str(daily_dates[0].date()),
        "end_date":    str(daily_dates[-1].date()),
        "n_days":      n_days,
        "top_k":       k,
        "exposure_note": "dollar-neutral: 0.5*(long_ret - short_ret)",
        "longshort_gross": {
            "total_return_pct":   round(total_return_ls  * 100, 4),
            "ann_return_pct":     round(ann_return_ls    * 100, 4),
            "volatility_ann_pct": round(vol_ls           * 100, 4),
            "sharpe":             round(sharpe_ls,              4),
            "sortino":            round(sortino_ls,             4),
            "max_drawdown_pct":   round(max_dd_ls        * 100, 4),
            "calmar":             round(calmar_ls,              4),
            "win_rate_pct":       round(win_rate_ls      * 100, 4),
        },
        "turnover": {
            "avg_daily_pct":   round(avg_turnover * 100, 2),
            "ann_turnover_x":  round(ann_turnover,       2),
        },
        "cost_sensitivity": cost_table,
        "long_only_gross": {
            "total_return_pct":   round(total_return_long * 100, 4),
            "ann_return_pct":     round(ann_return_long   * 100, 4),
            "sharpe":             round(sharpe_long,             4),
            "win_rate_pct":       round(win_rate_long     * 100, 4),
        },
        "ic": {
            ik: (round(v, 4) if not np.isnan(v) else None)
            for ik, v in ic_stats.items()
        },
        "quintile": {
            q: round(float(np.mean(v)) * 252 * 100, 4)
            for q, v in q_returns.items()
        },
        "q1_q5_spread_ann_pct": round(q_spread * 252 * 100, 4),
        "yearly": {str(yr): d for yr, d in yearly.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    # ---- Plots ----
    ckpt_suffix = f"  [{ckpt_path.stem}]"
    _plot_equity_curve(
        daily_dates, ls_ret, long_ret, short_ret,
        ic_arr, ic_stats["ic_mean"], k,
        out_dir / "equity_curve.png",
        title_suffix=ckpt_suffix,
    )
    _plot_quintiles(q_returns, out_dir / "quintile_returns.png")
    _plot_ic_histogram(ic_arr, ic_stats["ic_mean"], out_dir / "ic_histogram.png")
    _plot_cost_sensitivity(cost_table, out_dir / "cost_sensitivity.png")
    _plot_monthly_heatmap(daily_dates, ls_ret, out_dir / "monthly_heatmap.png")

    # Raw series for compare_models.py
    ic_df = pd.DataFrame({"date": daily_dates, "ic": ic_arr, "ls_ret": ls_ret})
    ic_df.to_csv(out_dir / "ic_timeseries.csv", index=False)
    metrics_json["ic_series"]     = [round(float(x), 6) if not np.isnan(x) else None for x in ic_arr]
    metrics_json["ls_ret_series"] = [round(float(x), 8) for x in ls_ret]
    metrics_json["dates"]         = [str(d.date()) for d in daily_dates]
    (out_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    print(f"\nResults saved to: {out_dir}")
    print(f"  metrics_report.txt  metrics.json  ic_timeseries.csv")
    print(f"  equity_curve.png  quintile_returns.png  ic_histogram.png")
    print(f"  monthly_heatmap.png  cost_sensitivity.png")


def _plot_monthly_heatmap(
    dates: list[pd.Timestamp],
    daily_returns: np.ndarray,
    out_path: Path,
) -> None:
    s = pd.Series(daily_returns, index=pd.DatetimeIndex(dates))
    # Compound monthly returns
    monthly = (s + 1).resample("ME").prod() - 1
    df = monthly.to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot(index="year", columns="month", values="ret") * 100
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, max(3, len(pivot) * 0.8 + 1)))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.01)
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Monthly L-S Returns (%) - Hybrid THGNN x MaGNet Mamba+MoE")
    plt.colorbar(im, ax=ax, label="%", fraction=0.02, pad=0.04)

    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                        fontsize=7.5, color="black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
