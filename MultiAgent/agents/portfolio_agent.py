"""
Portfolio Constructor Agent — fuses GNN predicted return rankings with news
sentiment scores into a final ranked buy/sell candidate list.

Fusion formula:
    final_score = alpha * gnn_rank + (1 - alpha) * sentiment_score

where gnn_rank is the GNN output cross-sectionally standardised to [0, 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rank_normalise(scores: dict[str, float]) -> dict[str, float]:
    """Map raw scores to percentile ranks in [0, 1]. Rank 1 = highest score."""
    if not scores:
        return {}
    tickers = list(scores.keys())
    vals = np.array([scores[t] for t in tickers], dtype=float)
    # Percentile rank: fraction of values below each value
    ranks = vals.argsort().argsort().astype(float) / max(len(vals) - 1, 1)
    return {t: float(r) for t, r in zip(tickers, ranks)}


def _zscore_normalise(scores: dict[str, float]) -> dict[str, float]:
    """Z-score normalise then clip to [-2, 2] and shift to [0, 1]."""
    if not scores:
        return {}
    tickers = list(scores.keys())
    vals = np.array([scores[t] for t in tickers], dtype=float)
    if vals.std() == 0:
        return {t: 0.5 for t in tickers}
    zs = (vals - vals.mean()) / vals.std()
    zs = np.clip(zs, -2, 2)
    normalised = (zs + 2) / 4  # map [-2, 2] → [0, 1]
    return {t: float(n) for t, n in zip(tickers, normalised)}


def construct(
    gnn_signals: dict[str, float],
    news_signals: dict[str, dict],
    alpha: float = 0.7,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    gnn_signals  : {ticker: raw_predicted_return}
    news_signals : {ticker: {"sentiment_score": float, "news_count": int, ...}}
    alpha        : weight on GNN signal (1 - alpha on sentiment)
    top_k        : number of long + short candidates to surface

    Returns
    -------
    DataFrame with columns: ticker, gnn_rank, sentiment_score, news_count,
                            final_score, action
    sorted by final_score descending.
    """
    common = set(gnn_signals.keys()) & set(news_signals.keys())
    if not common:
        # Fall back to GNN-only if no overlap
        common = set(gnn_signals.keys())

    gnn_rank = _rank_normalise({t: gnn_signals[t] for t in common if t in gnn_signals})

    sentiment_raw = {
        t: news_signals[t]["sentiment_score"]
        for t in common
        if t in news_signals and isinstance(news_signals[t], dict)
    }
    # For tickers with no news, treat sentiment as neutral (0.5 after normalisation)
    for t in common:
        if t not in sentiment_raw:
            sentiment_raw[t] = 0.0
    sentiment_norm = _zscore_normalise(sentiment_raw)

    rows = []
    for ticker in sorted(common):
        gr = gnn_rank.get(ticker, 0.5)
        sn = sentiment_norm.get(ticker, 0.5)
        nc = news_signals.get(ticker, {}).get("news_count", 0) if ticker in news_signals else 0
        sr = sentiment_raw.get(ticker, 0.0)

        if alpha == 1.0 or nc == 0:
            final = gr
        else:
            final = alpha * gr + (1 - alpha) * sn

        rows.append({
            "ticker":          ticker,
            "gnn_rank":        round(gr, 4),
            "sentiment_score": round(sr, 4),
            "sentiment_norm":  round(sn, 4),
            "news_count":      nc,
            "final_score":     round(final, 4),
        })

    df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)

    # Assign actions
    n = len(df)
    k = min(top_k, n // 2)
    actions = ["HOLD"] * n
    for i in range(k):
        actions[i] = "BUY"
    for i in range(n - k, n):
        actions[i] = "SELL"
    df["action"] = actions

    return df
