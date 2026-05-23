"""
compare_models.py — Run all three model variants and produce thesis-ready comparison plots.

Loads THGNN, Hybrid BiGRU, and Mamba-MoE checkpoints, runs inference on the same
holdout period, and generates four publication-quality comparison figures:

  1. comparison_cumulative_ls.png     — Overlaid cumulative L-S return curves
  2. comparison_rolling_ic.png        — Overlaid 60-day rolling IC curves
  3. comparison_quintile_returns.png  — Grouped quintile bar chart (3 models × 5 quintiles)
  4. comparison_yearly_ic.png         — Grouped yearly IC bar chart

Plus a summary CSV and a combined multi-panel figure.

Usage
-----
    # Run all three from scratch
    python compare_models.py --start-date 2024-01-01 --end-date 2026-05-15

    # Load from pre-saved backtest JSON files (much faster — skips inference)
    python compare_models.py \\
        --thgnn-json THGNN/data/backtest_results/thgnn_backtest_xxx/metrics.json \\
        --bigru-json THGNN/data/backtest_results/hybrid_backtest_xxx/metrics.json \\
        --mamba-json THGNN/data/backtest_results/mamba_backtest_xxx/metrics.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

# Checkpoints were saved on Python 3.12 Linux (PosixPath lives at pathlib._local.PosixPath).
# Two fixes needed for Python 3.11 Windows:
#   1. Register a fake pathlib._local so the class can be imported.
#   2. Remap PosixPath → PurePosixPath so instances can be created on Windows.
def _patch_pathlib_for_linux_ckpts() -> None:
    import types, pathlib as _pl
    # Fix 1: pathlib._local shim
    if "pathlib._local" not in sys.modules:
        shim = types.ModuleType("pathlib._local")
        shim.PosixPath   = _pl.PurePosixPath   # type: ignore[attr-defined]
        shim.WindowsPath = _pl.PureWindowsPath  # type: ignore[attr-defined]
        sys.modules["pathlib._local"] = shim
    # Fix 2: remap PosixPath in the main pathlib module so instantiation works
    if hasattr(_pl, "PosixPath"):
        _pl.PosixPath = _pl.PurePosixPath      # type: ignore[attr-defined]

_patch_pathlib_for_linux_ckpts()

ROOT           = Path(__file__).resolve().parent
THGNN_DIR      = ROOT / "THGNN"
BIGRU_DIR      = ROOT / "THGNN_MaGNet"
MAMBA_DIR      = ROOT / "THGNN_Mamba_MoE"
TRAIN_DATA_DIR = THGNN_DIR / "data" / "data_train_predict"
MODEL_DIR      = THGNN_DIR / "data" / "model_saved"
OUT_DIR        = ROOT / "comparison_results"

# Colours and labels consistent across all plots
MODEL_META = {
    "thgnn": {"label": "Base THGNN",   "color": "#e74c3c", "ls": "--"},
    "mamba": {"label": "Mamba--MoE",   "color": "#f39c12", "ls": "-."},
    "bigru": {"label": "Hybrid BiGRU", "color": "#2980b9", "ls": "-"},
}

# Insert THGNN on sys.path once — needed for AllGraphDataSampler and extract_data.
# Model classes are loaded via importlib (no sys.path needed).
if str(THGNN_DIR) not in sys.path:
    sys.path.insert(0, str(THGNN_DIR))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare all three GNN variants on holdout data.")
    p.add_argument("--start-date", type=str, default="2024-01-01")
    p.add_argument("--end-date",   type=str, default="2026-05-15")
    p.add_argument("--top-k",      type=int, default=5)
    p.add_argument("--device",     type=str, default=None)
    p.add_argument("--out-dir",    type=str, default=None)
    p.add_argument("--thgnn-json", type=str, default=None,
                   help="Path to metrics.json from a previous thgnn_backtest run.")
    p.add_argument("--bigru-json", type=str, default=None,
                   help="Path to metrics.json from a previous hybrid_backtest run.")
    p.add_argument("--mamba-json", type=str, default=None,
                   help="Path to metrics.json from a previous mamba_backtest run.")
    p.add_argument("--thgnn-ckpt", type=str, default=None)
    p.add_argument("--bigru-ckpt", type=str, default=None)
    p.add_argument("--mamba-ckpt", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Module loading helpers (avoids sys.path namespace collisions)
# ---------------------------------------------------------------------------

def _load_class_from_file(unique_name: str, py_file: Path, class_name: str):
    """Import a class from an arbitrary .py file without touching sys.path."""
    spec = importlib.util.spec_from_file_location(unique_name, py_file)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod          # register so internal imports resolve
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def _load_thgnn_class():
    return _load_class_from_file(
        "thgnn_model", THGNN_DIR / "model" / "Thgnn.py", "StockHeteGAT"
    )


def _load_bigru_class():
    return _load_class_from_file(
        "bigru_model", BIGRU_DIR / "model" / "hybrid_model.py", "HybridStockModel"
    )


def _load_mamba_class():
    return _load_class_from_file(
        "mamba_model", MAMBA_DIR / "model" / "hybrid_model.py", "HybridStockModel"
    )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _auto_ckpt(glob: str) -> Path:
    candidates = sorted(MODEL_DIR.glob(glob), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint matching '{glob}' in {MODEL_DIR}")
    return candidates[-1]


def _load_ckpt(path: Path, device: torch.device) -> dict:
    """Load checkpoint. The pathlib._local shim registered at module load handles
    checkpoints saved on Python 3.12 (where PosixPath lives in pathlib._local)."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _date_index(file_dates: list[pd.Timestamp], date_str: str, mode: str = "start") -> int:
    dt = pd.to_datetime(date_str).normalize()
    if mode == "start":
        for i, d in enumerate(file_dates):
            if d >= dt:
                return i
        return len(file_dates) - 1
    for i, d in enumerate(file_dates):
        if d > dt:
            return i
    return len(file_dates)


def _get_file_dates(start_date: str, end_date: str):
    files = sorted([p.name for p in TRAIN_DATA_DIR.glob("*.pkl")])
    if not files:
        raise RuntimeError(f"No .pkl files in {TRAIN_DATA_DIR}")
    dates = [pd.to_datetime(Path(n).stem).normalize() for n in files]
    s = _date_index(dates, start_date, "start")
    e = min(_date_index(dates, end_date, "end"), len(files))
    return dates, s, e


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_thgnn(ckpt: dict, device: torch.device):
    StockHeteGAT = _load_thgnn_class()
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg   = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = StockHeteGAT(
        in_features         = int(cfg.get("in_features",  12)),
        hidden_dim          = int(cfg.get("hidden_dim",   128)),
        num_heads           = int(cfg.get("num_heads",    4)),
        num_layers          = int(cfg.get("num_layers",   1)),
        out_features        = int(cfg.get("out_features", 32)),
        predictor_out_dim   = state["predictor.0.weight"].shape[0],
        predictor_activation= None,
        dropout             = 0.0,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _build_hybrid(ckpt: dict, device: torch.device, loader_fn):
    HybridStockModel = loader_fn()
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg   = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    def _i(k, d): return int(cfg.get(k, d))
    model = HybridStockModel(
        in_features         = _i("in_features",         12),
        embed_dim           = _i("embed_dim",            64),
        num_mage_layers     = _i("num_mage_layers",       1),
        num_moe_experts     = _i("num_moe_experts",       4),
        num_mha_heads       = _i("num_mha_heads",         2),
        gat_heads           = _i("gat_heads",             8),
        gat_out_features    = _i("gat_out_features",      8),
        num_hyper_edges     = _i("num_hyper_edges",      32),
        num_tch_hyper_edges = _i("num_tch_hyper_edges",  32),
        num_tch_heads       = _i("num_tch_heads",         4),
        dropout             = 0.0,
        predictor_out_dim   = 1,
        predictor_activation= None,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _extract_hybrid(data: dict, device: torch.device):
    pos_adj  = data["pos_adj"].to(device)
    neg_adj  = data["neg_adj"].to(device)
    features = data["features"].to(device)
    labels   = data["labels"].to(device)
    def _sq(t, mn):
        while t.dim() > mn and t.size(0) == 1:
            t = t.squeeze(0)
        return t
    features = _sq(features, 3)
    pos_adj  = _sq(pos_adj,  2)
    neg_adj  = _sq(neg_adj,  2)
    if labels.dim() > 1 and labels.size(-1) == 1:
        labels = labels.squeeze(-1)
    mask = data["mask"]
    if isinstance(mask, torch.Tensor):
        mask = mask.to(device).bool()
    else:
        mask = torch.tensor(mask, device=device, dtype=torch.bool)
    return pos_adj, neg_adj, features, labels, mask


def _run_inference(model, extract_fn, file_dates, start_idx: int, end_idx: int,
                   top_k: int, label: str) -> dict:
    from data_loader import AllGraphDataSampler

    ds = AllGraphDataSampler(
        str(TRAIN_DATA_DIR), mode="val",
        data_start=0, data_middle=start_idx, data_end=end_idx,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, pin_memory=False, collate_fn=lambda x: x
    )

    preds_all:  list[np.ndarray]   = []
    actual_all: list[np.ndarray]   = []
    dates_all:  list[pd.Timestamp] = []

    for i, batch in enumerate(tqdm(loader, desc=f"[{label}]")):
        data = batch[0]
        try:
            pos_adj, neg_adj, features, labels, mask = extract_fn(data)
        except Exception:
            continue
        with torch.no_grad():
            p = model(features, pos_adj, neg_adj).squeeze(-1)

        p_np = p[mask].cpu().numpy()
        a_np = labels[mask].cpu().numpy()
        if a_np.ndim > 1:
            a_np = a_np[:, 0]
        if len(p_np) < top_k * 2:
            continue

        preds_all.append(p_np)
        actual_all.append(a_np)
        dates_all.append(file_dates[start_idx + i])

    return _aggregate(preds_all, actual_all, dates_all, top_k, label)


def _aggregate(preds, actuals, dates, top_k, label) -> dict:
    n = len(preds)
    ls_ret   = np.empty(n)
    long_ret = np.empty(n)
    for t, (p, a) in enumerate(zip(preds, actuals)):
        order       = np.argsort(p)[::-1]
        long_ret[t] = a[order[:top_k]].mean()
        ls_ret[t]   = 0.5 * (a[order[:top_k]].mean() - a[order[-top_k:]].mean())

    ic_arr = np.array([
        stats.spearmanr(p, a)[0] if len(p) >= 5 else float("nan")
        for p, a in zip(preds, actuals)
    ], dtype=float)

    q_returns: dict[str, list[float]] = {f"Q{i+1}": [] for i in range(5)}
    for p, a in zip(preds, actuals):
        nn = len(p)
        if nn < 10:
            continue
        order = np.argsort(p)[::-1]
        for q in range(5):
            lo = q * nn // 5
            hi = (q + 1) * nn // 5
            q_returns[f"Q{q+1}"].append(float(a[order[lo:hi]].mean()))

    dates_idx = pd.DatetimeIndex(dates)
    yearly_ic: dict[int, float] = {}
    for yr in sorted(set(dates_idx.year)):
        sub = ic_arr[dates_idx.year == yr]
        valid = sub[~np.isnan(sub)]
        yearly_ic[yr] = float(np.mean(valid)) if len(valid) > 0 else float("nan")

    valid_ic   = ic_arr[~np.isnan(ic_arr)]
    ic_mean    = float(np.mean(valid_ic)) if len(valid_ic) else float("nan")
    ic_std     = float(np.std(valid_ic))  if len(valid_ic) else float("nan")
    icir       = (ic_mean / ic_std) if ic_std > 0 else float("nan")
    sharpe     = (float(ls_ret.mean() / ls_ret.std() * np.sqrt(252))
                  if ls_ret.std() > 0 else float("nan"))
    ann_factor = 252 / n
    cum_ls     = np.cumprod(1 + ls_ret) - 1
    total_ret  = float(cum_ls[-1])
    ann_ret    = float((1 + total_ret) ** ann_factor - 1)
    equity     = np.cumprod(1 + ls_ret)
    peak       = np.maximum.accumulate(equity)
    max_dd     = float(((equity - peak) / peak).min())

    return {
        "label":       label,
        "dates":       dates,
        "ic_arr":      ic_arr,
        "ls_ret":      ls_ret,
        "q_returns":   q_returns,
        "yearly_ic":   yearly_ic,
        "ic_mean":     ic_mean,
        "icir":        icir,
        "sharpe":      sharpe,
        "ann_return":  ann_ret,
        "max_dd":      max_dd,
        "total_return": total_ret,
        "n_days":      n,
    }


# ---------------------------------------------------------------------------
# Load from pre-saved JSON
# ---------------------------------------------------------------------------

def _load_from_json(json_path: str, label: str) -> dict:
    with open(json_path) as f:
        d = json.load(f)
    dates  = [pd.to_datetime(dt) for dt in d["dates"]]
    ic_arr = np.array([x if x is not None else float("nan") for x in d["ic_series"]], dtype=float)
    ls_ret = np.array(d["ls_ret_series"], dtype=float)

    yearly_ic: dict[int, float] = {}
    if "yearly" in d:
        for yr_str, yd in d["yearly"].items():
            yearly_ic[int(yr_str)] = float(yd.get("mean_ic", float("nan")))

    q_returns: dict[str, list[float]] = {f"Q{i+1}": [] for i in range(5)}
    if "quintile" in d:
        for q, ann_ret_pct in d["quintile"].items():
            # Convert annualised % back to a single daily-equivalent mean
            q_returns[q] = [ann_ret_pct / (252 * 100)]

    valid_ic  = ic_arr[~np.isnan(ic_arr)]
    ic_mean   = float(np.mean(valid_ic)) if len(valid_ic) else float("nan")
    ic_std    = float(np.std(valid_ic))  if len(valid_ic) else float("nan")
    icir      = (ic_mean / ic_std) if ic_std > 0 else float("nan")
    ls_stats  = d.get("longshort_gross", {})

    return {
        "label":        label,
        "dates":        dates,
        "ic_arr":       ic_arr,
        "ls_ret":       ls_ret,
        "q_returns":    q_returns,
        "yearly_ic":    yearly_ic,
        "ic_mean":      ic_mean,
        "icir":         icir,
        "sharpe":       ls_stats.get("sharpe", float("nan")),
        "ann_return":   ls_stats.get("ann_return_pct", float("nan")),
        "max_dd":       ls_stats.get("max_drawdown_pct", float("nan")),
        "total_return": ls_stats.get("total_return_pct", float("nan")),
        "n_days":       d.get("n_days", len(dates)),
    }


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def plot_cumulative_ls(results: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    for r in results:
        m   = MODEL_META[r["key"]]
        cum = np.cumprod(1 + r["ls_ret"]) - 1
        ax.plot(r["dates"], cum * 100, color=m["color"], lw=2, ls=m["ls"],
                label=f"{m['label']}  (Sharpe = {r['sharpe']:.2f})")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Cumulative L-S Return (%)")
    ax.set_title("Holdout Cumulative Return — Equal-Weight Long-Short  (top-5 / bottom-5, dollar-neutral)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def plot_rolling_ic(results: list[dict], out_path: Path, window: int = 60) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    for r in results:
        m   = MODEL_META[r["key"]]
        ic_s = pd.Series(r["ic_arr"], index=r["dates"])
        rolling = ic_s.rolling(window, min_periods=window // 3).mean()
        ax.plot(r["dates"], rolling, color=m["color"], lw=2, ls=m["ls"],
                label=f"{m['label']}  (mean IC = {r['ic_mean']:.4f})")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_ylabel(f"Spearman Rank-IC ({window}-day rolling mean)")
    ax.set_title(f"{window}-Day Rolling Rank-IC — All Model Variants on Holdout Set")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def plot_quintile_comparison(results: list[dict], out_path: Path) -> None:
    quintiles = [f"Q{i+1}" for i in range(5)]
    x         = np.arange(len(quintiles))
    n_models  = len(results)
    width     = 0.22
    offsets   = np.linspace(-(n_models - 1) * width / 2,
                             (n_models - 1) * width / 2, n_models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, r in zip(offsets, results):
        m     = MODEL_META[r["key"]]
        means = [np.mean(r["q_returns"].get(q, [0.0])) * 252 * 100 for q in quintiles]
        ax.bar(x + offset, means, width, color=m["color"],
               alpha=0.85, label=m["label"], edgecolor="white")

    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "Q1\n(top pred.)", "Q2", "Q3", "Q4", "Q5\n(bot. pred.)"
    ])
    ax.set_ylabel("Annualised Return (%)")
    ax.set_title("Quintile Return Analysis — Monotone Q1→Q5 spread validates rank signal\n"
                 "(equal-weight within each quintile, daily rebalancing)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def plot_yearly_ic(results: list[dict], out_path: Path) -> None:
    all_years = sorted({yr for r in results for yr in r["yearly_ic"].keys()})
    x         = np.arange(len(all_years))
    n_models  = len(results)
    width     = 0.22
    offsets   = np.linspace(-(n_models - 1) * width / 2,
                             (n_models - 1) * width / 2, n_models)

    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, r in zip(offsets, results):
        m       = MODEL_META[r["key"]]
        ic_vals = [r["yearly_ic"].get(yr, float("nan")) for yr in all_years]
        bars    = ax.bar(x + offset, ic_vals, width, color=m["color"],
                         alpha=0.85, label=m["label"], edgecolor="white")
        for bar, val in zip(bars, ic_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + (0.0005 if val >= 0 else -0.002),
                        f"{val:.4f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{yr}" + (" (partial)" if yr == max(all_years) else "")
                        for yr in all_years])
    ax.set_ylabel("Mean Spearman Rank-IC")
    ax.set_title("Per-Year Rank-IC on Holdout Set")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def plot_combined(results: list[dict], out_path: Path, window: int = 60) -> None:
    """Single 2×2 figure for thesis inclusion."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Model Comparison — Holdout Set (2024-01-01 → 2026-05-15)", fontsize=13)

    # Top-left: cumulative L-S
    ax = axes[0, 0]
    for r in results:
        m   = MODEL_META[r["key"]]
        cum = np.cumprod(1 + r["ls_ret"]) - 1
        ax.plot(r["dates"], cum * 100, color=m["color"], lw=2, ls=m["ls"], label=m["label"])
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_title("(a) Cumulative L-S Return (%)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Top-right: rolling IC
    ax = axes[0, 1]
    for r in results:
        m    = MODEL_META[r["key"]]
        ic_s = pd.Series(r["ic_arr"], index=r["dates"])
        ax.plot(r["dates"], ic_s.rolling(window, min_periods=window // 3).mean(),
                color=m["color"], lw=2, ls=m["ls"], label=m["label"])
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_title(f"(b) {window}-Day Rolling Rank-IC")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Bottom-left: quintiles
    ax = axes[1, 0]
    quintiles = [f"Q{i+1}" for i in range(5)]
    x = np.arange(5)
    n = len(results)
    width   = 0.22
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)
    for offset, r in zip(offsets, results):
        m     = MODEL_META[r["key"]]
        means = [np.mean(r["q_returns"].get(q, [0.0])) * 252 * 100 for q in quintiles]
        ax.bar(x + offset, means, width, color=m["color"], alpha=0.85,
               label=m["label"], edgecolor="white")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(["Q1\n(top)", "Q2", "Q3", "Q4", "Q5\n(bot.)"])
    ax.set_title("(c) Quintile Annualised Return (%)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # Bottom-right: yearly IC
    ax = axes[1, 1]
    all_years = sorted({yr for r in results for yr in r["yearly_ic"].keys()})
    x2 = np.arange(len(all_years))
    for offset, r in zip(offsets, results):
        m       = MODEL_META[r["key"]]
        ic_vals = [r["yearly_ic"].get(yr, float("nan")) for yr in all_years]
        ax.bar(x2 + offset, ic_vals, width, color=m["color"], alpha=0.85,
               label=m["label"], edgecolor="white")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(x2)
    ax.set_xticklabels([f"{yr}" + (" *" if yr == max(all_years) else "")
                        for yr in all_years])
    ax.set_title("(d) Per-Year Mean Rank-IC  (* = partial year)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def save_summary_csv(results: list[dict], out_path: Path) -> None:
    rows = []
    for r in results:
        rows.append({
            "Model":              MODEL_META[r["key"]]["label"],
            "Mean Rank-IC":       round(r["ic_mean"],   4),
            "ICIR":               round(r["icir"],      4) if not np.isnan(r["icir"]) else "nan",
            "Sharpe (gross)":     round(r["sharpe"],    4) if not np.isnan(r["sharpe"]) else "nan",
            "Ann Return % (gross)": round(r["ann_return"], 2) if not np.isnan(r["ann_return"]) else "nan",
            "Max Drawdown %":     round(r["max_dd"],    2) if not np.isnan(r["max_dd"]) else "nan",
            "N Days":             r["n_days"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n{'=' * 65}")
    print(df.to_string(index=False))
    print(f"{'=' * 65}")
    print(f"  {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results_raw: dict[str, dict] = {}

    # ---- Load from JSON if provided ----
    if args.thgnn_json:
        print("Loading THGNN results from JSON...")
        results_raw["thgnn"] = _load_from_json(args.thgnn_json, "thgnn")
    if args.bigru_json:
        print("Loading BiGRU results from JSON...")
        results_raw["bigru"] = _load_from_json(args.bigru_json, "bigru")
    if args.mamba_json:
        print("Loading Mamba results from JSON...")
        results_raw["mamba"] = _load_from_json(args.mamba_json, "mamba")

    # ---- Run inference for any model not loaded from JSON ----
    need_inference = any(k not in results_raw for k in ("thgnn", "bigru", "mamba"))
    if need_inference:
        file_dates, s_idx, e_idx = _get_file_dates(args.start_date, args.end_date)
        print(f"Backtesting {e_idx - s_idx} days: "
              f"{file_dates[s_idx].date()} -> {file_dates[e_idx - 1].date()}")
        from trainer.trainer import extract_data  # THGNN-compatible extractor

        def _extract_thgnn(data):
            return extract_data(data, device)

        def _extract_hybrid_fn(data):
            return _extract_hybrid(data, device)

    if "thgnn" not in results_raw:
        try:
            ckpt_path = Path(args.thgnn_ckpt) if args.thgnn_ckpt else _auto_ckpt("*_icrank_best.dat")
            ckpt      = _load_ckpt(ckpt_path, device)
            model     = _build_thgnn(ckpt, device)
            epoch     = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
            print(f"[THGNN] Loaded {ckpt_path.name}  (epoch {epoch})")
            results_raw["thgnn"] = _run_inference(
                model, _extract_thgnn, file_dates, s_idx, e_idx, args.top_k, "THGNN"
            )
        except Exception as ex:
            print(f"[THGNN] Failed: {ex}")

    if "bigru" not in results_raw:
        try:
            ckpt_path = Path(args.bigru_ckpt) if args.bigru_ckpt else _auto_ckpt("*_hybrid_best.dat")
            ckpt      = _load_ckpt(ckpt_path, device)
            model     = _build_hybrid(ckpt, device, _load_bigru_class)
            epoch     = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
            print(f"[BiGRU] Loaded {ckpt_path.name}  (epoch {epoch})")
            results_raw["bigru"] = _run_inference(
                model, _extract_hybrid_fn, file_dates, s_idx, e_idx, args.top_k, "Hybrid BiGRU"
            )
        except Exception as ex:
            print(f"[BiGRU] Failed: {ex}")

    if "mamba" not in results_raw:
        try:
            ckpt_path = Path(args.mamba_ckpt) if args.mamba_ckpt else _auto_ckpt("*_mamba_moe_best.dat")
            ckpt      = _load_ckpt(ckpt_path, device)
            model     = _build_hybrid(ckpt, device, _load_mamba_class)
            epoch     = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
            print(f"[Mamba] Loaded {ckpt_path.name}  (epoch {epoch})")
            results_raw["mamba"] = _run_inference(
                model, _extract_hybrid_fn, file_dates, s_idx, e_idx, args.top_k, "Mamba-MoE"
            )
        except Exception as ex:
            print(f"[Mamba] Failed: {ex}")

    if not results_raw:
        print("No models ran successfully. Exiting.")
        return

    # Tag with model key and use canonical order
    for k, r in results_raw.items():
        r["key"] = k
    ordered  = [k for k in ("thgnn", "mamba", "bigru") if k in results_raw]
    results  = [results_raw[k] for k in ordered]

    # ---- Generate all plots ----
    print("\nGenerating comparison plots:")
    plot_cumulative_ls      (results, out_dir / "comparison_cumulative_ls.png")
    plot_rolling_ic         (results, out_dir / "comparison_rolling_ic.png")
    plot_quintile_comparison(results, out_dir / "comparison_quintile_returns.png")
    plot_yearly_ic          (results, out_dir / "comparison_yearly_ic.png")
    plot_combined           (results, out_dir / "comparison_combined.png")
    save_summary_csv        (results, out_dir / "comparison_summary.csv")

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
