"""
alpha_sweep.py — Sensitivity of portfolio composition to fusion weight alpha.

Uses the 25-May-2026 portfolio.csv (21 NSE stocks, Hybrid BiGRU inference +
FinBERT sentiment) to show how the top-K BUY and bottom-K SELL sets, and the
Spearman rank-correlation between fusion score and GNN-only score, change
as alpha varies from 0.0 to 1.0.

Produces:
  - alpha_sweep_table.csv        (for the thesis table)
  - alpha_sweep_topk.csv         (ticker membership at each alpha)
  - comparison_results/alpha_sweep_rank_corr.png
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parent
PORT    = ROOT / "portfolio.csv"
OUT_DIR = ROOT / "comparison_results"
OUT_DIR.mkdir(exist_ok=True)

# ── Load portfolio ────────────────────────────────────────────────────────────
df = pd.read_csv(PORT)
df["ticker"] = df["ticker"].str.replace(r"\.NS$", "", regex=True)

# Alpha-sweep logic mirrors stock_analysis.py:
#   • If news_count == 0  →  score = gnn_rank  (no sentiment blending)
#   • If news_count  > 0  →  score = alpha*gnn_rank + (1-alpha)*sentiment_norm
def fusion(alpha: float, row: pd.Series) -> float:
    if row["news_count"] == 0:
        if alpha == 1.0:
            return float(row["gnn_rank"])
        elif alpha == 0.0:
            return float(row["sentiment_norm"])   # neutral 0.1828 for all no-news stocks
        else:
            return float(row["gnn_rank"])         # no-news → gnn-only at any alpha > 0
    return float(alpha * row["gnn_rank"] + (1 - alpha) * row["sentiment_norm"])

ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
K = 5   # top-K BUY / bottom-K SELL (consistent with backtest)

# α=1.0 reference (pure GNN)
ref_scores = np.array([fusion(1.0, r) for _, r in df.iterrows()])
ref_rank   = stats.rankdata(ref_scores)   # ascending rank
ref_top5   = set(df["ticker"][np.argsort(ref_scores)[::-1][:K]])
ref_bot5   = set(df["ticker"][np.argsort(ref_scores)[:K]])

rows = []
membership_rows = []

for alpha in ALPHAS:
    scores = np.array([fusion(alpha, r) for _, r in df.iterrows()])
    order_desc = np.argsort(scores)[::-1]
    order_asc  = np.argsort(scores)

    top5  = set(df["ticker"].iloc[order_desc[:K]])
    bot5  = set(df["ticker"].iloc[order_asc[:K]])

    # Spearman ρ between this alpha's scores and GNN-only scores
    rho, pval = stats.spearmanr(scores, ref_scores)

    # Count membership changes vs alpha=1.0
    buy_changes  = len(ref_top5 - top5)   # stocks dropped from BUY
    sell_changes = len(ref_bot5 - bot5)   # stocks dropped from SELL

    # Q1-Q5 spread using actual returns is not available; report rank-only metrics
    rows.append({
        "alpha":       alpha,
        "rho_vs_gnn":  round(rho, 4),
        "buy_changes": buy_changes,
        "sell_changes":sell_changes,
        "top5_buy":    ", ".join(df["ticker"].iloc[order_desc[:K]].tolist()),
        "bot5_sell":   ", ".join(df["ticker"].iloc[order_asc[:K]].tolist()),
    })

    membership_rows.append({
        "alpha": alpha,
        "BUY_1": df["ticker"].iloc[order_desc[0]],
        "BUY_2": df["ticker"].iloc[order_desc[1]],
        "BUY_3": df["ticker"].iloc[order_desc[2]],
        "BUY_4": df["ticker"].iloc[order_desc[3]],
        "BUY_5": df["ticker"].iloc[order_desc[4]],
        "SELL_1": df["ticker"].iloc[order_asc[0]],
        "SELL_2": df["ticker"].iloc[order_asc[1]],
        "SELL_3": df["ticker"].iloc[order_asc[2]],
        "SELL_4": df["ticker"].iloc[order_asc[3]],
        "SELL_5": df["ticker"].iloc[order_asc[4]],
    })

summary = pd.DataFrame(rows)
membership = pd.DataFrame(membership_rows)

summary.to_csv(OUT_DIR / "alpha_sweep_table.csv", index=False)
membership.to_csv(OUT_DIR / "alpha_sweep_topk.csv", index=False)

print("\n=== Alpha Sweep Summary ===")
print(summary[["alpha","rho_vs_gnn","buy_changes","sell_changes"]].to_string(index=False))
print("\n=== Top-5 BUY per alpha ===")
for _, r in summary.iterrows():
    print(f"  α={r['alpha']:.1f}: {r['top5_buy']}")
print("\n=== Bottom-5 SELL per alpha ===")
for _, r in summary.iterrows():
    print(f"  α={r['alpha']:.1f}: {r['bot5_sell']}")

# ── Plot: rank-correlation vs alpha ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot([r["alpha"] for r in rows],
        [r["rho_vs_gnn"] for r in rows],
        "o-", color="#2980b9", lw=2, ms=7)
for r in rows:
    ax.annotate(f'{r["rho_vs_gnn"]:.3f}',
                xy=(r["alpha"], r["rho_vs_gnn"]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9)
ax.set_xlabel("Fusion weight α (GNN share)", fontsize=11)
ax.set_ylabel("Spearman ρ vs GNN-only ranking", fontsize=11)
ax.set_title("Alpha sensitivity: rank correlation vs pure GNN (25 May 2026)", fontsize=11)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0.0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "alpha_sweep_rank_corr.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → {OUT_DIR / 'alpha_sweep_rank_corr.png'}")
