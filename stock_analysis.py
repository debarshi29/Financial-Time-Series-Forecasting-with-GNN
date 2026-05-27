"""
stock_analysis.py — Stock-level analysis of GNN buy recommendations (thesis supplement).

Loads all three model checkpoints (THGNN, Hybrid BiGRU, Mamba-MoE), runs inference
on the holdout period, captures per-day long/short stock picks, maps indices to NSE
ticker symbols, and generates eight publication-quality figures that complement the
four panels already produced by compare_models.py:

  1. long_stock_frequency.png      — Top-20 most recommended stocks, grouped by model
  2. sector_allocation.png         — Quarterly sector composition of long portfolios
  3. model_agreement_jaccard.png   — Rolling Jaccard similarity between model pairs
  4. consensus_portfolio.png       — 2- and 3-model consensus vs solo equity curves
  5. hit_rate_by_agreement.png     — Win rate + mean return by number of agreeing models
  6. top_stock_trajectories.png    — Cumulative return paths of top-10 recommended stocks
  7. return_dispersion.png         — Long (Q1) vs short (Q5) actual return distributions
  8. predicted_rank_vs_return.png  — Decile return profile (rank-IC signal validation)

Plus a per-stock CSV summary: frequency, hit rate, and mean return when long.

Usage
-----
    # Full inference run
    python stock_analysis.py --start-date 2024-01-01 --end-date 2026-05-15

    # Cache results so you can re-generate plots without re-running inference
    python stock_analysis.py --cache inference_cache.pkl

    # Next time — load from cache (fast)
    python stock_analysis.py --cache inference_cache.pkl

    # Specific checkpoints
    python stock_analysis.py \\
        --thgnn-ckpt THGNN/data/model_saved/my_thgnn.dat \\
        --bigru-ckpt THGNN/data/model_saved/my_bigru.dat \\
        --mamba-ckpt THGNN/data/model_saved/my_mamba.dat
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm


# ---------------------------------------------------------------------------
# pathlib shim — same fix as compare_models.py for Linux-saved checkpoints
# ---------------------------------------------------------------------------

def _patch_pathlib_for_linux_ckpts() -> None:
    import types, pathlib as _pl
    if "pathlib._local" not in sys.modules:
        shim = types.ModuleType("pathlib._local")
        shim.PosixPath   = _pl.PurePosixPath
        shim.WindowsPath = _pl.PureWindowsPath
        sys.modules["pathlib._local"] = shim
    if hasattr(_pl, "PosixPath"):
        _pl.PosixPath = _pl.PurePosixPath

_patch_pathlib_for_linux_ckpts()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT           = Path(__file__).resolve().parent
THGNN_DIR      = ROOT / "THGNN"
BIGRU_DIR      = ROOT / "THGNN_MaGNet"
MAMBA_DIR      = ROOT / "THGNN_Mamba_MoE"
TRAIN_DATA_DIR = THGNN_DIR / "data" / "data_train_predict"
MODEL_DIR      = THGNN_DIR / "data" / "model_saved"
TICKERS_FILE   = THGNN_DIR / "data" / "valid_nifty500.txt"
SECTORS_FILE   = THGNN_DIR / "ind_nifty500list.csv"
OUT_DIR        = ROOT / "stock_analysis_results"

# Insert THGNN on sys.path once for AllGraphDataSampler + extract_data
if str(THGNN_DIR) not in sys.path:
    sys.path.insert(0, str(THGNN_DIR))


# ---------------------------------------------------------------------------
# Visual constants — kept consistent with compare_models.py
# ---------------------------------------------------------------------------

MODEL_META = {
    "thgnn": {"label": "Base THGNN",   "color": "#e74c3c", "ls": "--"},
    "mamba": {"label": "Mamba–MoE",    "color": "#f39c12", "ls": "-."},
    "bigru": {"label": "Hybrid BiGRU", "color": "#2980b9", "ls": "-"},
}

# Sector palette — enough for 14+ GICS-like categories
_SECTOR_PAL = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#795548",
    "#607d8b", "#00bcd4", "#8bc34a", "#ff5722", "#ffd700",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-stock analysis of GNN buy recommendations for thesis."
    )
    p.add_argument("--start-date",     default="2024-01-01")
    p.add_argument("--end-date",       default="2026-05-15")
    p.add_argument("--top-k",          type=int, default=5)
    p.add_argument("--device",         default=None)
    p.add_argument("--out-dir",        default=None)
    p.add_argument("--thgnn-ckpt",     default=None)
    p.add_argument("--bigru-ckpt",     default=None)
    p.add_argument("--mamba-ckpt",     default=None)
    p.add_argument("--cache",          default=None,
                   help="Pickle file to save/load per-day inference results. "
                        "Loaded if it exists; saved after a fresh inference run.")
    p.add_argument("--rolling-window", type=int, default=30,
                   help="Rolling window (days) for Jaccard similarity plot.")
    p.add_argument("--top-n-stocks",   type=int, default=20,
                   help="Number of stocks shown in the frequency bar chart.")
    p.add_argument("--top-n-traj",     type=int, default=10,
                   help="Number of stocks shown in the trajectory chart.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Symbol / sector helpers
# ---------------------------------------------------------------------------

def _load_ticker_list() -> list[str]:
    """Load valid_nifty500.txt → ordered list of NSE ticker symbols."""
    if not TICKERS_FILE.exists():
        print(f"  Warning: {TICKERS_FILE} not found — will use positional indices.")
        return []
    lines = [ln.strip() for ln in TICKERS_FILE.read_text().splitlines() if ln.strip()]
    return lines


def _load_sector_map() -> dict[str, str]:
    """Load ind_nifty500list.csv → {symbol_without_suffix: sector_string}."""
    if not SECTORS_FILE.exists():
        return {}
    try:
        df = pd.read_csv(SECTORS_FILE)
        sym_col = next((c for c in df.columns if "symbol" in c.lower()), None)
        ind_col = next((c for c in df.columns
                        if "industry" in c.lower() or "sector" in c.lower()), None)
        if sym_col is None or ind_col is None:
            return {}
        return {str(sym).strip(): str(ind).strip()
                for sym, ind in zip(df[sym_col], df[ind_col])}
    except Exception as exc:
        print(f"  Warning: could not parse sector file: {exc}")
        return {}


def _ticker_sector(ticker: str, sector_map: dict) -> str:
    """Map 'RELIANCE.NS' → sector label (or 'Other')."""
    sym = ticker.replace(".NS", "").replace(".BO", "").strip()
    return sector_map.get(sym, "Other")


# ---------------------------------------------------------------------------
# Module loading helpers (avoids sys.path namespace collisions)
# ---------------------------------------------------------------------------

def _load_class(unique_name: str, py_file: Path, class_name: str):
    spec = importlib.util.spec_from_file_location(unique_name, py_file)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)

def _cls_thgnn():
    return _load_class("thgnn_model", THGNN_DIR / "model" / "Thgnn.py", "StockHeteGAT")

def _cls_bigru():
    return _load_class("bigru_model", BIGRU_DIR / "model" / "hybrid_model.py", "HybridStockModel")

def _cls_mamba():
    return _load_class("mamba_model", MAMBA_DIR / "model" / "hybrid_model.py", "HybridStockModel")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _auto_ckpt(glob_pat: str) -> Path:
    candidates = sorted(MODEL_DIR.glob(glob_pat), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint matching '{glob_pat}' in {MODEL_DIR}")
    return candidates[-1]


def _load_ckpt(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_thgnn(ckpt: dict, device: torch.device):
    StockHeteGAT = _cls_thgnn()
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
# Data extraction — compatible with both model families
# ---------------------------------------------------------------------------

def _extract_thgnn(data: dict, device: torch.device):
    from trainer.trainer import extract_data
    return extract_data(data, device)


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


# ---------------------------------------------------------------------------
# Per-day inference with stock-level capture
# ---------------------------------------------------------------------------

def _run_inference(
    model,
    extract_fn,
    file_dates: list[pd.Timestamp],
    start_idx: int,
    end_idx:   int,
    top_k:     int,
    label:     str,
    device:    torch.device,
    all_tickers: list[str],
) -> list[dict]:
    """
    Returns a list of dicts, one per valid trading day:
      date          — pd.Timestamp
      tickers       — list[str]  (all active stocks that day, ordered)
      pred          — np.ndarray (predicted scores for active stocks)
      actual        — np.ndarray (actual next-day return for active stocks)
      pred_ranks    — np.ndarray (rank percentile 0-1 for each active stock)
      long_tickers  — list[str]  (top-K predicted stocks)
      short_tickers — list[str]  (bottom-K predicted stocks)
      long_actual   — np.ndarray (actual returns for long picks)
      short_actual  — np.ndarray (actual returns for short picks)
    """
    from data_loader import AllGraphDataSampler

    ds = AllGraphDataSampler(
        str(TRAIN_DATA_DIR), mode="val",
        data_start=0, data_middle=start_idx, data_end=end_idx,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, pin_memory=False, collate_fn=lambda x: x
    )

    daily: list[dict] = []
    n_total = len(all_tickers)

    for i, batch in enumerate(tqdm(loader, desc=f"[{label}]")):
        data = batch[0]
        try:
            pos_adj, neg_adj, features, labels, mask = extract_fn(data, device)
        except Exception as exc:
            print(f"  [{label}] skip day {i}: {exc}")
            continue

        with torch.no_grad():
            preds = model(features, pos_adj, neg_adj).squeeze(-1)

        # Normalise mask to bool tensor on the right device
        if isinstance(mask, torch.Tensor):
            mask_t = mask.to(preds.device).bool()
        else:
            mask_t = torch.tensor(mask, device=preds.device, dtype=torch.bool)

        mask_np = mask_t.cpu().numpy().astype(bool)

        p = preds[mask_t].cpu().numpy()
        a = labels[mask_t].cpu().numpy()
        if a.ndim > 1:
            a = a[:, 0]   # take t+1 horizon

        if len(p) < top_k * 2:
            continue

        # --- Map mask → active tickers for this day ---
        if n_total > 0 and len(mask_np) == n_total:
            # Full-universe mask: mask_np[j] == True → all_tickers[j] is active
            day_tickers = [all_tickers[j] for j in range(n_total) if mask_np[j]]
        elif n_total > 0 and len(mask_np) == len(p):
            # Already masked: all entries are active
            day_tickers = list(all_tickers[: len(p)])
        else:
            # Fallback: positional labels
            day_tickers = [f"STK_{j}" for j in range(len(p))]

        # Safety clip (shouldn't be needed but guards against off-by-one)
        min_len = min(len(p), len(day_tickers))
        p = p[:min_len]
        a = a[:min_len]
        day_tickers = day_tickers[:min_len]

        # Rank percentile: 0 = worst predicted, 1 = best predicted
        pred_ranks = stats.rankdata(p).astype(float) / len(p)

        # Top-K / bottom-K selection
        order      = np.argsort(p)[::-1]
        long_idx   = order[:top_k]
        short_idx  = order[-top_k:]

        daily.append({
            "date":          file_dates[start_idx + i],
            "tickers":       day_tickers,
            "pred":          p,
            "actual":        a,
            "pred_ranks":    pred_ranks,
            "long_tickers":  [day_tickers[j] for j in long_idx],
            "short_tickers": [day_tickers[j] for j in short_idx],
            "long_actual":   a[long_idx],
            "short_actual":  a[short_idx],
        })

    return daily


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _build_stock_stats(daily: list[dict], label: str) -> pd.DataFrame:
    """Per-stock: times in long portfolio, hit rate, mean long return."""
    freq     : dict[str, int]   = defaultdict(int)
    hits     : dict[str, int]   = defaultdict(int)
    ret_sum  : dict[str, float] = defaultdict(float)

    for d in daily:
        for t, a in zip(d["long_tickers"], d["long_actual"]):
            freq[t]    += 1
            ret_sum[t] += float(a)
            if float(a) > 0:
                hits[t] += 1

    rows = []
    for t, f in freq.items():
        rows.append({
            "ticker":       t,
            "model":        label,
            "long_count":   f,
            "hit_rate":     hits[t] / f,
            "avg_long_ret": ret_sum[t] / f,
        })

    return pd.DataFrame(rows).sort_values("long_count", ascending=False)


def _stock_daily_returns(daily: list[dict]) -> dict[str, pd.Series]:
    """Return daily actual-return series for every active stock in the period."""
    ret_dict: dict[str, dict] = defaultdict(dict)
    for d in daily:
        for t, a in zip(d["tickers"], d["actual"]):
            ret_dict[t][d["date"]] = float(a)
    return {t: pd.Series(rv).sort_index() for t, rv in ret_dict.items()}


def _jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else float("nan")


def _common_dates(daily_dict: dict[str, list[dict]]) -> pd.DatetimeIndex:
    """Timestamps present in every model's inference output."""
    date_sets = [set(d["date"] for d in v) for v in daily_dict.values()]
    common = date_sets[0]
    for s in date_sets[1:]:
        common &= s
    return pd.DatetimeIndex(sorted(common))


def _cum(arr: np.ndarray) -> np.ndarray:
    return np.cumprod(1 + arr) - 1


# ---------------------------------------------------------------------------
# Figure 1: Long-portfolio stock frequency
# ---------------------------------------------------------------------------

def plot_stock_frequency(
    all_stats: dict[str, pd.DataFrame],
    out_path: Path,
    top_n: int = 20,
) -> None:
    """
    Grouped bar chart: for the top-N most recommended stocks (across all models),
    show how many times each model placed that stock in its long portfolio.
    Tells the reader which names the models consistently favour.
    """
    agg      = pd.concat(all_stats.values())[["ticker", "long_count"]]
    top_tkrs = agg.groupby("ticker")["long_count"].sum().nlargest(top_n).index.tolist()

    keys    = list(all_stats.keys())
    n       = len(keys)
    width   = 0.22
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)
    x       = np.arange(len(top_tkrs))

    fig, ax = plt.subplots(figsize=(14, 5))
    for offset, key in zip(offsets, keys):
        df   = all_stats[key].set_index("ticker")
        vals = [int(df.loc[t, "long_count"]) if t in df.index else 0 for t in top_tkrs]
        ax.bar(x + offset, vals, width, color=MODEL_META[key]["color"],
               alpha=0.85, label=MODEL_META[key]["label"], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(".NS", "") for t in top_tkrs],
                       rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("Days in Long Portfolio")
    ax.set_title(
        f"Top-{top_n} Most Frequently Recommended Stocks — Long Portfolio\n"
        "(equal-weight, daily rebalancing, holdout set)"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 2: Quarterly sector allocation
# ---------------------------------------------------------------------------

def plot_sector_allocation(
    daily_dict: dict[str, list[dict]],
    sector_map: dict,
    out_path:   Path,
) -> None:
    """
    Stacked bar chart per model: what fraction of each model's long picks come from
    each GICS sector, aggregated by calendar quarter. Shows whether models exhibit
    sector concentration or diversified coverage.
    """
    keys = list(daily_dict.keys())

    # Collect sector universe
    all_sectors: set = set()
    for key in keys:
        for d in daily_dict[key]:
            for t in d["long_tickers"]:
                all_sectors.add(_ticker_sector(t, sector_map))
    all_sectors_list = sorted(all_sectors)
    sec_color = {s: _SECTOR_PAL[i % len(_SECTOR_PAL)]
                 for i, s in enumerate(all_sectors_list)}

    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 5), sharey=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        rows = []
        for d in daily_dict[key]:
            dt  = d["date"]
            qtr = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
            for t in d["long_tickers"]:
                rows.append({"qtr": qtr, "sector": _ticker_sector(t, sector_map)})
        if not rows:
            ax.set_title(MODEL_META[key]["label"])
            continue

        pivot = (pd.DataFrame(rows)
                   .groupby(["qtr", "sector"]).size()
                   .unstack(fill_value=0))
        pivot = pivot.div(pivot.sum(axis=1), axis=0).sort_index()

        bottom = np.zeros(len(pivot))
        for sec in all_sectors_list:
            if sec not in pivot.columns:
                continue
            vals = pivot[sec].values
            ax.bar(range(len(pivot)), vals, bottom=bottom,
                   color=sec_color[sec], edgecolor="white", linewidth=0.4)
            bottom += vals

        ax.set_title(MODEL_META[key]["label"], fontsize=11)
        ax.set_xticks(range(len(pivot)))
        ax.set_xticklabels(pivot.index.tolist(), rotation=45, ha="right", fontsize=7.5)
        ax.set_ylim(0, 1)

    handles = [Patch(color=sec_color[s], label=s) for s in all_sectors_list]
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(all_sectors_list)),
               fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Quarterly Sector Composition of Long Portfolio", fontsize=12, y=1.01)
    axes[0].set_ylabel("Fraction of long picks")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 3: Pairwise model agreement (Jaccard similarity)
# ---------------------------------------------------------------------------

def plot_model_agreement(
    daily_dict: dict[str, list[dict]],
    out_path:   Path,
    window:     int = 30,
) -> None:
    """
    Two-panel figure:
      Top:    Rolling Jaccard similarity of long-portfolio overlap for each model pair.
      Bottom: Rolling fraction of long picks where all three models agree.
    High agreement → natural ensemble validation; Jaccard spread → model diversity.
    """
    keys = list(daily_dict.keys())
    if len(keys) < 2:
        print("  [model_agreement] need ≥ 2 models — skipping.")
        return

    cdates = _common_dates(daily_dict)
    if len(cdates) < window:
        print("  [model_agreement] too few common dates — skipping.")
        return

    long_sets: dict[str, dict] = {
        k: {d["date"]: set(d["long_tickers"]) for d in v}
        for k, v in daily_dict.items()
    }

    # Pairwise combinations
    pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]
    pair_jac: dict[tuple, list] = {p: [] for p in pairs}
    all_agree: list[float]      = []

    for dt in cdates:
        today = [long_sets[k].get(dt, set()) for k in keys]
        for pa in pairs:
            pair_jac[pa].append(_jaccard(long_sets[pa[0]].get(dt, set()),
                                         long_sets[pa[1]].get(dt, set())))
        inter = today[0]
        union = today[0]
        for s in today[1:]:
            inter = inter & s
            union = union | s
        all_agree.append(len(inter) / len(union) if union else 0.0)

    # Pair colours — cycle through model colours
    pair_colors = ["#8e44ad", "#27ae60", "#c0392b"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    for (ka, kb), col in zip(pairs, pair_colors):
        s       = pd.Series(pair_jac[(ka, kb)], index=cdates)
        rolling = s.rolling(window, min_periods=window // 3).mean()
        lbl     = f"{MODEL_META[ka]['label']} ∩ {MODEL_META[kb]['label']}"
        axes[0].plot(cdates, rolling, color=col, lw=2, label=lbl)

    axes[0].set_ylabel(f"Jaccard Similarity ({window}-day rolling mean)")
    axes[0].set_title("Pairwise Long-Portfolio Overlap Between Models")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    s_all   = pd.Series(all_agree, index=cdates)
    r_all   = s_all.rolling(window, min_periods=window // 3).mean()
    axes[1].fill_between(cdates, r_all * 100, alpha=0.35, color="#2ecc71")
    axes[1].plot(cdates, r_all * 100, color="#27ae60", lw=2)
    axes[1].set_ylabel(f"All-Model Jaccard (%) — {window}-day rolling")
    axes[1].set_title("Three-Model Consensus Overlap Rate")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 4: Consensus portfolio equity curves
# ---------------------------------------------------------------------------

def plot_consensus_portfolio(
    daily_dict: dict[str, list[dict]],
    out_path:   Path,
) -> None:
    """
    Overlaid cumulative return curves for:
      • Each model's solo long portfolio (top-K stocks)
      • 2-model consensus: stocks in the long portfolio of at least 2 models
      • 3-model consensus: stocks in the long portfolio of all 3 models
    Tests whether ensemble agreement adds signal beyond any single model.
    Directly motivates the 'Ensemble and stacking' future-work direction.
    """
    keys   = list(daily_dict.keys())
    cdates = _common_dates(daily_dict)
    if len(cdates) < 5:
        print("  [consensus_portfolio] too few common dates — skipping.")
        return

    long_sets:   dict[str, dict] = {k: {d["date"]: set(d["long_tickers"]) for d in v}
                                     for k, v in daily_dict.items()}
    actual_maps: dict[str, dict] = {k: {d["date"]: dict(zip(d["tickers"], d["actual"]))
                                        for d in v} for k, v in daily_dict.items()}

    solo_ret : dict[str, list] = {k: [] for k in keys}
    cons2_ret: list[float]     = []
    cons3_ret: list[float]     = []

    for dt in cdates:
        ref = actual_maps[keys[0]].get(dt, {})

        for key in keys:
            picks = long_sets[key].get(dt, set())
            solo_ret[key].append(
                float(np.mean([actual_maps[key].get(dt, {}).get(t, 0.0) for t in picks]))
                if picks else 0.0
            )

        sets_today = [long_sets[k].get(dt, set()) for k in keys]

        # 2-model consensus: union of pairwise intersections
        cons2 = set()
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                cons2 |= sets_today[i] & sets_today[j]
        cons2_ret.append(
            float(np.mean([ref.get(t, 0.0) for t in cons2])) if cons2 else 0.0
        )

        # 3-model consensus: intersection of all
        cons3 = sets_today[0]
        for s in sets_today[1:]:
            cons3 = cons3 & s
        cons3_ret.append(
            float(np.mean([ref.get(t, 0.0) for t in cons3])) if cons3 else 0.0
        )

    dates_arr = list(cdates)
    fig, ax   = plt.subplots(figsize=(13, 5))

    for key in keys:
        m = MODEL_META[key]
        ax.plot(dates_arr, _cum(np.array(solo_ret[key])) * 100,
                color=m["color"], lw=1.6, ls=m["ls"], label=m["label"], alpha=0.75)

    ax.plot(dates_arr, _cum(np.array(cons2_ret)) * 100,
            color="#8e44ad", lw=2.5, ls="-", label="2-Model Consensus", zorder=4)
    ax.plot(dates_arr, _cum(np.array(cons3_ret)) * 100,
            color="#27ae60", lw=2.5, ls="-", label="3-Model Consensus", zorder=5)

    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Cumulative Long-Only Return (%)")
    ax.set_title(
        "Consensus Portfolio Returns — Long Positions Where Models Agree\n"
        "(2-model: any two agree; 3-model: all three agree; equal-weight daily rebalance)"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 5: Hit rate by number of agreeing models
# ---------------------------------------------------------------------------

def plot_hit_rate_by_agreement(
    daily_dict: dict[str, list[dict]],
    out_path:   Path,
) -> None:
    """
    For every (date, ticker) that appears in at least one long portfolio, record how
    many models selected it and whether the actual return was positive.
    Bar charts: win rate (%) and mean daily return (%) by agreement count.
    Directly tests the value of ensemble agreement as a confidence filter.
    """
    keys = list(daily_dict.keys())
    cdates_set = set(_common_dates(daily_dict))

    long_sets:   dict[str, dict] = {k: {d["date"]: set(d["long_tickers"]) for d in v}
                                     for k, v in daily_dict.items()}
    actual_maps: dict[str, dict] = {k: {d["date"]: dict(zip(d["tickers"], d["actual"]))
                                        for d in v} for k, v in daily_dict.items()}

    records: list[dict] = []
    for dt in cdates_set:
        # Union of all long picks today
        union_long = set()
        for key in keys:
            union_long |= long_sets[key].get(dt, set())

        ref = actual_maps[keys[0]].get(dt, {})
        for t in union_long:
            n_agree = sum(1 for k in keys if t in long_sets[k].get(dt, set()))
            actual  = ref.get(t, float("nan"))
            if not np.isnan(actual):
                records.append({"n_agree": n_agree, "actual": actual})

    if not records:
        print("  [hit_rate] no records — skipping.")
        return

    df = pd.DataFrame(records)
    groups   = sorted(df["n_agree"].unique())
    g_labels = [f"{g} model{'s' if g > 1 else ''}" for g in groups]
    colors   = ["#e74c3c", "#f39c12", "#27ae60", "#3498db"][:len(groups)]

    hit_rates = [100 * (df[df.n_agree == g]["actual"] > 0).mean() for g in groups]
    avg_rets  = [100 * df[df.n_agree == g]["actual"].mean() for g in groups]
    counts    = [len(df[df.n_agree == g]) for g in groups]

    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    bars0 = axes[0].bar(x, hit_rates, color=colors, edgecolor="white", alpha=0.85)
    for bar, c in zip(bars0, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"n={c:,}", ha="center", va="bottom", fontsize=8.5)
    axes[0].axhline(50, color="gray", lw=1.2, ls="--", label="Random (50%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(g_labels, fontsize=10)
    axes[0].set_ylabel("Win Rate (%)")
    axes[0].set_title("Long Position Win Rate\nby Number of Agreeing Models")
    axes[0].set_ylim(0, max(100, max(hit_rates) + 5))
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    bars1 = axes[1].bar(x, avg_rets, color=colors, edgecolor="white", alpha=0.85)
    for bar, c in zip(bars1, counts):
        y_off = bar.get_height() + 0.001 if bar.get_height() >= 0 else bar.get_height() - 0.008
        axes[1].text(bar.get_x() + bar.get_width() / 2, y_off,
                     f"n={c:,}", ha="center", va="bottom", fontsize=8.5)
    axes[1].axhline(0, color="gray", lw=0.8, ls=":")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(g_labels, fontsize=10)
    axes[1].set_ylabel("Mean Daily Return (%)")
    axes[1].set_title("Mean Long Position Return\nby Number of Agreeing Models")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Ensemble Agreement as a Confidence Filter\n"
        "(stocks agreed upon by more models show higher win rate and mean return)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 6: Cumulative return trajectories for top recommended stocks
# ---------------------------------------------------------------------------

def plot_top_stock_trajectories(
    daily_dict:  dict[str, list[dict]],
    all_stats:   dict[str, pd.DataFrame],
    sector_map:  dict,
    out_path:    Path,
    top_n:       int = 10,
) -> None:
    """
    For the top-N stocks by combined recommendation frequency across all models,
    plot the cumulative return path (buy-and-hold from day 1 of the holdout period)
    derived from the actual daily returns stored in the pkl labels.
    Annotated with final return % and colour-coded by sector.
    """
    agg       = pd.concat(all_stats.values())[["ticker", "long_count"]]
    top_tkrs  = agg.groupby("ticker")["long_count"].sum().nlargest(top_n).index.tolist()

    # Use the model with most inference days (usually all equal; prefer bigru)
    key = "bigru" if "bigru" in daily_dict else list(daily_dict.keys())[0]
    daily = daily_dict[key]

    stock_rets = _stock_daily_returns(daily)

    sectors = [_ticker_sector(t, sector_map) for t in top_tkrs]
    uniq_s  = list(dict.fromkeys(sectors))
    s_color = {s: _SECTOR_PAL[i % len(_SECTOR_PAL)] for i, s in enumerate(uniq_s)}

    fig, ax = plt.subplots(figsize=(13, 6))
    plotted = 0

    for t, sec in zip(top_tkrs, sectors):
        if t not in stock_rets or len(stock_rets[t]) < 5:
            continue
        ret_s = stock_rets[t]
        cum   = _cum(ret_s.values) * 100
        sym   = t.replace(".NS", "")
        ax.plot(ret_s.index, cum, lw=1.8, color=s_color[sec],
                label=f"{sym} ({sec})", alpha=0.85)
        # Final-value annotation offset
        ax.annotate(
            f"{cum[-1]:.1f}%",
            xy=(ret_s.index[-1], cum[-1]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=7.5, color=s_color[sec], va="center",
        )
        plotted += 1

    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(
        f"Cumulative Return Paths — Top-{plotted} Most Frequently Recommended Stocks\n"
        f"(buy-and-hold from holdout start; returns from {MODEL_META[key]['label']} labels)"
    )
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 7: Return distribution — long (Q1) vs short (Q5)
# ---------------------------------------------------------------------------

def plot_return_dispersion(
    daily_dict: dict[str, list[dict]],
    out_path:   Path,
) -> None:
    """
    Violin plots (with median markers) comparing actual next-day returns for
    long picks (top-K) versus short picks (bottom-K) across all models.
    A well-calibrated rank signal should show long > short in both mean and median.
    Complements the quintile bar chart in compare_models.py with full distributions.
    """
    keys = list(daily_dict.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 5), sharey=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        long_rets  = np.concatenate([d["long_actual"]  for d in daily_dict[key]]) * 100
        short_rets = np.concatenate([d["short_actual"] for d in daily_dict[key]]) * 100

        # Clip at 99th percentile for visual clarity
        clip = float(np.percentile(np.abs(np.concatenate([long_rets, short_rets])), 99))
        long_rets  = np.clip(long_rets,  -clip, clip)
        short_rets = np.clip(short_rets, -clip, clip)

        try:
            parts = ax.violinplot(
                [long_rets, short_rets], positions=[0, 1],
                showmedians=True, showextrema=False,
            )
            parts["bodies"][0].set_facecolor("#2ecc71")
            parts["bodies"][0].set_alpha(0.60)
            parts["bodies"][1].set_facecolor("#e74c3c")
            parts["bodies"][1].set_alpha(0.60)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(2)
        except Exception:
            ax.boxplot([long_rets, short_rets], positions=[0, 1])

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Long (Q1)\nTop-K picks", "Short (Q5)\nBottom-K picks"])
        ax.set_title(MODEL_META[key]["label"])
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.grid(True, alpha=0.2, axis="y")

        # Annotate means
        for pos, data, col, va in [
            (0, long_rets,  "#27ae60", "bottom"),
            (1, short_rets, "#c0392b", "top"),
        ]:
            ax.text(pos, data.mean(), f"μ = {data.mean():.3f}%",
                    ha="center", va=va, fontsize=9,
                    color=col, fontweight="bold")

    axes[0].set_ylabel("Actual Next-Day Return (%)")
    fig.suptitle(
        "Distribution of Actual Returns — Long vs Short Portfolio Picks\n"
        "(long picks should be right-shifted relative to short picks → confirms rank signal)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 8: Predicted rank decile vs actual return
# ---------------------------------------------------------------------------

def plot_predicted_rank_vs_return(
    daily_dict: dict[str, list[dict]],
    out_path:   Path,
    n_bins:     int = 10,
) -> None:
    """
    For each model, bin all (stock, day) pairs into N deciles by predicted rank
    and compute the mean actual return per decile.  A well-calibrated rank-IC
    signal should show a monotone D1 → D10 increase (or at least a clear spread),
    matching the quintile analysis but at finer resolution and for all three models.
    Error bars show ±1 standard error of the mean.
    """
    keys = list(daily_dict.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 5), sharey=True)
    if len(keys) == 1:
        axes = [axes]

    bin_labels = [f"D{i + 1}" for i in range(n_bins)]

    for ax, key in zip(axes, keys):
        bins  = [[] for _ in range(n_bins)]
        for d in daily_dict[key]:
            for rank, act in zip(d["pred_ranks"], d["actual"]):
                idx = min(int(rank * n_bins), n_bins - 1)
                bins[idx].append(act)

        means  = [100 * np.mean(b) if b else 0.0 for b in bins]
        sems   = [100 * np.std(b) / np.sqrt(max(len(b), 1)) for b in bins]
        colors = ["#e74c3c" if m < 0 else "#2ecc71" for m in means]

        ax.bar(range(n_bins), means, color=colors, edgecolor="white", alpha=0.85)
        ax.errorbar(range(n_bins), means, yerr=sems,
                    fmt="none", color="black", capsize=3, linewidth=1.2)
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels(bin_labels, fontsize=8.5)
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("Predicted Return Decile\n(D1 = lowest predicted, D10 = highest)")
        ax.set_title(MODEL_META[key]["label"])
        ax.grid(True, alpha=0.2, axis="y")

        # Overlay trend line
        xs = np.arange(n_bins)
        try:
            slope, intercept, *_ = stats.linregress(xs, means)
            ax.plot(xs, intercept + slope * xs, color="black",
                    lw=1.5, ls="--", alpha=0.6, label=f"Trend (slope={slope:.3f})")
            ax.legend(fontsize=8)
        except Exception:
            pass

    axes[0].set_ylabel("Mean Actual Next-Day Return (%)")
    fig.suptitle(
        "Rank-IC Validation — Predicted Decile vs Realised Return\n"
        "(monotone D1→D10 spread confirms the rank signal; error bars = ±1 SE)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 9: Daily portfolio turnover
# ---------------------------------------------------------------------------

def plot_portfolio_turnover(
    daily_dict: dict[str, list[dict]],
    out_path:   Path,
    window:     int = 20,
) -> None:
    """
    For each model, compute daily turnover as the fraction of the long portfolio
    that changes from day t-1 to day t:
        turnover_t = 1 - |long_t ∩ long_{t-1}| / |long_t ∪ long_{t-1}|
    A value of 1.0 means every position was replaced; 0.0 means no change.
    Also plots mean daily turnover as an annotation.
    Directly contextualises the reported Sharpe ratios: high turnover in a
    daily-rebalancing equal-weight portfolio means transaction costs matter.
    """
    fig, ax = plt.subplots(figsize=(13, 4))

    summary_rows = []
    for key in daily_dict:
        m      = MODEL_META[key]
        daily  = daily_dict[key]
        dates  = [d["date"]          for d in daily]
        lsets  = [set(d["long_tickers"]) for d in daily]

        turnover = [float("nan")]   # undefined for the first day
        for i in range(1, len(lsets)):
            union = lsets[i] | lsets[i - 1]
            inter = lsets[i] & lsets[i - 1]
            turnover.append(1.0 - len(inter) / len(union) if union else float("nan"))

        to_s    = pd.Series(turnover, index=dates)
        rolling = to_s.rolling(window, min_periods=window // 3).mean()
        mean_to = float(to_s.dropna().mean())

        ax.plot(dates, rolling * 100, color=m["color"], lw=2, ls=m["ls"],
                label=f"{m['label']}  (mean={mean_to*100:.1f}%)")
        summary_rows.append({"model": m["label"], "mean_daily_turnover_pct": round(mean_to * 100, 2)})

    ax.set_ylabel(f"Daily Portfolio Turnover (%)  [{window}-day rolling]")
    ax.set_title(
        "Daily Long-Portfolio Turnover — Fraction of Positions Changed Each Day\n"
        "(100% = full replacement; low turnover = transaction costs less critical)"
    )
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")

    # Print summary
    print("  Turnover summary:")
    for r in summary_rows:
        print(f"    {r['model']:20s}  mean daily turnover = {r['mean_daily_turnover_pct']:.1f}%")


# ---------------------------------------------------------------------------
# Figure 10: Cost-adjusted Sharpe ratio sweep
# ---------------------------------------------------------------------------

def plot_cost_adjusted_sharpe(
    daily_dict: dict[str, list[dict]],
    out_path:   Path,
    costs_bps:  list[int] | None = None,
    top_k:      int = 5,
) -> None:
    """
    For each model, compute the net Sharpe ratio of the equal-weight L-S portfolio
    after subtracting round-trip transaction costs ranging from 0 to 30 bp per trade.

    Cost model: on each day, the top_k long and top_k short positions are traded.
    For simplicity every position is assumed to turn over (worst-case), so the
    daily cost deduction is:
        cost_t = cost_bps * 1e-4 * 2 * top_k / (2 * top_k)  = cost_bps * 1e-4
    (buying top_k at cost_bps each and selling top_k at cost_bps each, all
    normalised to a 1-unit gross exposure).

    The more realistic turnover-weighted cost (cost × daily_turnover_t) is also
    overlaid as a dashed line.
    """
    if costs_bps is None:
        costs_bps = [0, 3, 5, 8, 10, 15, 20, 25, 30]

    # Precompute turnover per day per model
    turnovers: dict[str, np.ndarray] = {}
    for key, daily in daily_dict.items():
        lsets = [set(d["long_tickers"]) for d in daily]
        to_arr = np.zeros(len(lsets))
        for i in range(1, len(lsets)):
            u = lsets[i] | lsets[i - 1]
            s = lsets[i] & lsets[i - 1]
            to_arr[i] = 1.0 - len(s) / len(u) if u else 1.0
        to_arr[0] = to_arr[1] if len(to_arr) > 1 else 1.0
        turnovers[key] = to_arr

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for key in daily_dict:
        m      = MODEL_META[key]
        daily  = daily_dict[key]
        to_arr = turnovers[key]

        # Raw L-S daily returns
        ls_ret = np.array([
            0.5 * (d["long_actual"].mean() - d["short_actual"].mean())
            for d in daily
        ])

        sharpes_worst = []
        sharpes_real  = []

        for c in costs_bps:
            c_frac = c * 1e-4
            # Worst case: full turnover every day
            net_worst = ls_ret - c_frac
            sh_w = net_worst.mean() / net_worst.std() * np.sqrt(252) if net_worst.std() > 0 else 0.0
            sharpes_worst.append(sh_w)
            # Realistic: cost proportional to actual turnover
            net_real = ls_ret - c_frac * to_arr
            sh_r = net_real.mean() / net_real.std() * np.sqrt(252) if net_real.std() > 0 else 0.0
            sharpes_real.append(sh_r)

        axes[0].plot(costs_bps, sharpes_worst, color=m["color"], lw=2, ls=m["ls"],
                     label=m["label"])
        axes[1].plot(costs_bps, sharpes_real,  color=m["color"], lw=2, ls=m["ls"],
                     label=m["label"])

    for ax, title in zip(axes, [
        "Worst-Case (100% daily turnover)",
        "Realistic (turnover-weighted cost)",
    ]):
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.axhline(1, color="gray", lw=0.6, ls=":", alpha=0.5)
        ax.set_xlabel("Round-Trip Transaction Cost (basis points)")
        ax.set_ylabel("Net Annualised Sharpe Ratio")
        ax.set_title(title)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(costs_bps[0], costs_bps[-1])

    fig.suptitle(
        "Cost-Adjusted Sharpe Ratio — Sensitivity to Transaction Costs\n"
        "(equal-weight L-S portfolio, daily rebalancing, top-5 / bottom-5)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 11: IC autocorrelation
# ---------------------------------------------------------------------------

def plot_ic_autocorrelation(
    daily_dict: dict[str, list[dict]],
    out_path:   Path,
    max_lag:    int = 20,
) -> None:
    """
    Plots the autocorrelation function (ACF) of the daily Spearman rank-IC series
    for each model up to max_lag days.

    Motivation: the reported t-statistics use the i.i.d. SE formula
        SE = 1 / sqrt(N_stocks - 2) / sqrt(T).
    If daily IC values are positively autocorrelated, the effective sample size
    T_eff < T and the true SE is larger — reported t-stats are upper bounds.
    This figure shows directly whether that concern is material.

    The 95% Bartlett confidence band (+/- 1.96/sqrt(T)) is overlaid in grey.
    """
    fig, axes = plt.subplots(1, len(daily_dict), figsize=(5.5 * len(daily_dict), 4),
                             sharey=True)
    if len(daily_dict) == 1:
        axes = [axes]

    for ax, key in zip(axes, daily_dict):
        m     = MODEL_META[key]
        daily = daily_dict[key]

        ic_arr = np.array([
            stats.spearmanr(d["pred"], d["actual"])[0]
            for d in daily
            if len(d["pred"]) >= 5
        ], dtype=float)
        ic_arr = ic_arr[~np.isnan(ic_arr)]
        T      = len(ic_arr)
        ic_c   = ic_arr - ic_arr.mean()   # demean before ACF

        lags = np.arange(0, max_lag + 1)
        acf  = np.array([
            float(np.corrcoef(ic_c[:T - l], ic_c[l:])[0, 1]) if l < T else 0.0
            for l in lags
        ])
        conf = 1.96 / np.sqrt(T)

        ax.bar(lags, acf, color=m["color"], alpha=0.75, edgecolor="white")
        ax.axhline(0,     color="black", lw=0.8)
        ax.axhline( conf, color="gray",  lw=1.2, ls="--", alpha=0.8, label="95% CI")
        ax.axhline(-conf, color="gray",  lw=1.2, ls="--", alpha=0.8)
        ax.fill_between([-0.5, max_lag + 0.5], [-conf, -conf], [conf, conf],
                        color="gray", alpha=0.10)
        ax.set_xlim(-0.5, max_lag + 0.5)
        ax.set_ylim(-0.35, 0.55)
        ax.set_xlabel("Lag (days)")
        ax.set_title(f"{m['label']}\n(T={T} days, mean IC={ic_arr.mean():.4f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

        # Print lag-1 autocorrelation
        rho1 = acf[1] if len(acf) > 1 else float("nan")
        t_eff = int(T * (1 - rho1) / (1 + rho1)) if abs(rho1) < 1 else 0
        print(f"  [{key:6s}] lag-1 ACF = {rho1:.3f}  T_eff ~ {t_eff}  "
              f"(vs i.i.d. T={T})")

    axes[0].set_ylabel("Autocorrelation")
    fig.suptitle(
        "IC Series Autocorrelation — Assessing i.i.d. Assumption for t-Statistics\n"
        "('95% CI' band: bars outside are significant; positive ACF inflates reported t-stats)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------

def save_stock_summary(
    all_stats: dict[str, pd.DataFrame],
    out_path:  Path,
) -> None:
    merged = pd.concat(all_stats.values()).sort_values(["ticker", "model"])
    merged["ticker_sym"] = merged["ticker"].str.replace(".NS", "", regex=False)
    merged = merged[["ticker_sym", "ticker", "model", "long_count", "hit_rate", "avg_long_ret"]]
    merged.columns = ["Symbol", "Ticker", "Model", "Long Count",
                      "Hit Rate", "Avg Long Return"]
    merged.to_csv(out_path, index=False, float_format="%.5f")

    # Quick console summary
    combined = merged.groupby("Symbol")["Long Count"].sum().nlargest(10)
    print(f"\n{'=' * 55}")
    print("Top-10 most recommended symbols (all models combined):")
    print(combined.to_string())
    print(f"{'=' * 55}")
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Symbol metadata ----
    all_tickers = _load_ticker_list()
    sector_map  = _load_sector_map()
    print(f"Tickers: {len(all_tickers)} loaded  |  Sectors: {len(sector_map)} mapped")

    # ---- Load or run inference ----
    cache_path = Path(args.cache) if args.cache else None

    if cache_path and cache_path.exists():
        print(f"\nLoading cached inference results: {cache_path}")
        with open(cache_path, "rb") as f:
            daily_dict: dict[str, list[dict]] = pickle.load(f)
        for k, v in daily_dict.items():
            print(f"  [{k:6s}] {len(v)} days loaded from cache")
    else:
        # Resolve date range → file indices
        data_files = sorted([p.name for p in TRAIN_DATA_DIR.glob("*.pkl")])
        if not data_files:
            raise RuntimeError(f"No .pkl files in {TRAIN_DATA_DIR}")
        file_dates = [pd.to_datetime(Path(n).stem).normalize() for n in data_files]

        def _idx(date_str: str, mode: str) -> int:
            dt = pd.to_datetime(date_str).normalize()
            for i, d in enumerate(file_dates):
                if mode == "start" and d >= dt:
                    return i
                if mode == "end" and d > dt:
                    return i
            return len(file_dates)

        s_idx = _idx(args.start_date, "start")
        e_idx = min(_idx(args.end_date, "end"), len(data_files))
        print(f"\nInference window: {e_idx - s_idx} days  "
              f"({file_dates[s_idx].date()} → {file_dates[e_idx - 1].date()})")

        daily_dict = {}

        def _try(key: str, ckpt_arg: str | None, glob_pat: str,
                 build_fn, extract_fn) -> None:
            try:
                cp = Path(ckpt_arg) if ckpt_arg else _auto_ckpt(glob_pat)
                ck = _load_ckpt(cp, device)
                md = build_fn(ck, device)
                epoch = ck.get("epoch", "?") if isinstance(ck, dict) else "?"
                print(f"\n[{key.upper()}] {cp.name}  (epoch {epoch})")
                daily_dict[key] = _run_inference(
                    md, extract_fn, file_dates, s_idx, e_idx,
                    args.top_k, key.upper(), device, all_tickers,
                )
                print(f"[{key.upper()}] {len(daily_dict[key])} valid days")
            except Exception as exc:
                print(f"\n[{key.upper()}] Failed: {exc}")

        _try("thgnn", args.thgnn_ckpt, "*_icrank_best.dat",
             _build_thgnn,
             _extract_thgnn)

        _try("bigru", args.bigru_ckpt, "*_hybrid_best.dat",
             lambda ck, dev: _build_hybrid(ck, dev, _cls_bigru),
             _extract_hybrid)

        _try("mamba", args.mamba_ckpt, "*_mamba_moe_best.dat",
             lambda ck, dev: _build_hybrid(ck, dev, _cls_mamba),
             _extract_hybrid)

        if not daily_dict:
            print("\nNo models ran successfully. Exiting.")
            return

        # Save cache
        if cache_path:
            print(f"\nSaving inference cache → {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(daily_dict, f)

    if not daily_dict:
        print("No inference data. Exiting.")
        return

    # ---- Per-stock statistics ----
    all_stats: dict[str, pd.DataFrame] = {}
    for key, daily in daily_dict.items():
        all_stats[key] = _build_stock_stats(daily, MODEL_META[key]["label"])

    # ---- Generate all figures ----
    print(f"\nGenerating stock-level analysis plots → {out_dir}")

    plot_stock_frequency(
        all_stats,
        out_dir / "long_stock_frequency.png",
        top_n=args.top_n_stocks,
    )

    plot_sector_allocation(
        daily_dict, sector_map,
        out_dir / "sector_allocation.png",
    )

    plot_model_agreement(
        daily_dict,
        out_dir / "model_agreement_jaccard.png",
        window=args.rolling_window,
    )

    plot_consensus_portfolio(
        daily_dict,
        out_dir / "consensus_portfolio.png",
    )

    plot_hit_rate_by_agreement(
        daily_dict,
        out_dir / "hit_rate_by_agreement.png",
    )

    plot_top_stock_trajectories(
        daily_dict, all_stats, sector_map,
        out_dir / "top_stock_trajectories.png",
        top_n=args.top_n_traj,
    )

    plot_return_dispersion(
        daily_dict,
        out_dir / "return_dispersion.png",
    )

    plot_predicted_rank_vs_return(
        daily_dict,
        out_dir / "predicted_rank_vs_return.png",
    )

    plot_portfolio_turnover(
        daily_dict,
        out_dir / "portfolio_turnover.png",
        window=20,
    )

    plot_cost_adjusted_sharpe(
        daily_dict,
        out_dir / "cost_adjusted_sharpe.png",
        top_k=args.top_k,
    )

    plot_ic_autocorrelation(
        daily_dict,
        out_dir / "ic_autocorrelation.png",
        max_lag=20,
    )

    save_stock_summary(
        all_stats,
        out_dir / "stock_analysis_summary.csv",
    )

    print(f"\nAll outputs saved to: {out_dir}")
    print(
        "\nFigures generated (add to thesis Results chapter):\n"
        "  long_stock_frequency.png     → which NIFTY 500 stocks are most recommended\n"
        "  sector_allocation.png        → quarterly sector exposure of long portfolio\n"
        "  model_agreement_jaccard.png  → pairwise / all-3 model agreement over time\n"
        "  consensus_portfolio.png      → 2/3-model consensus vs solo equity curves\n"
        "  hit_rate_by_agreement.png    → ensemble confidence: win rate by #models agreeing\n"
        "  top_stock_trajectories.png   → buy-and-hold return paths of top recommendations\n"
        "  return_dispersion.png        → long vs short actual return distributions\n"
        "  predicted_rank_vs_return.png → decile return profile (rank-IC validation)\n"
        "  portfolio_turnover.png       → daily % of positions changed per model\n"
        "  cost_adjusted_sharpe.png     → Sharpe vs transaction cost sweep (0-30bp)\n"
        "  ic_autocorrelation.png       → ACF of daily IC; validates t-stat assumptions\n"
        "  stock_analysis_summary.csv   → per-stock: frequency, hit rate, avg return\n"
    )


if __name__ == "__main__":
    main()
