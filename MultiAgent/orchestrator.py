"""
Multi-Agent Orchestrator — coordinates the three agents in a sequential pipeline:

  1. GNNAgent.predict(date)          → per-ticker predicted return scores
  2. NewsAgent.score_tickers(tickers) → per-ticker news sentiment scores (FinBERT)
  3. PortfolioAgent.construct(...)    → ranked buy/sell recommendation table

All agents are stateless except GNNAgent (which caches its loaded model).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from agents.gnn_agent import GNNAgent
from agents import news_agent as NewsAgent
from agents import portfolio_agent as PortfolioAgent


def run(
    date: str,
    top_k: int = 10,
    alpha: float = 0.7,
    max_news_headlines: int = 20,
    checkpoint_path: str | None = None,
    device: str | None = None,
    no_news: bool = False,
) -> pd.DataFrame:
    """
    Run the full multi-agent pipeline for a given date.

    Parameters
    ----------
    date            : Trading date string 'YYYY-MM-DD'
    top_k           : Top-K long + bottom-K short candidates to highlight
    alpha           : GNN signal weight (0=news-only, 1=GNN-only)
    max_news_headlines : Headlines per ticker fed to FinBERT
    checkpoint_path : Optional explicit model checkpoint path
    device          : 'cuda' or 'cpu' (auto-detected if None)
    no_news         : Skip news agent (GNN-only mode, faster for testing)

    Returns
    -------
    DataFrame sorted by final_score (best buy candidates first)
    """
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Multi-Agent Pipeline  |  date={date}  top_k={top_k}  alpha={alpha:.2f}")
    print(f"{'='*60}")

    # --- Agent 1: GNN predictions ---
    print(f"\n[Step 1/3] GNN Prediction Agent")
    agent = GNNAgent(checkpoint_path=checkpoint_path, device=device)
    gnn_signals = agent.predict(date)
    print(f"  -> {len(gnn_signals)} stocks with predictions")

    # --- Agent 2: News sentiment ---
    tickers = list(gnn_signals.keys())
    if no_news:
        print(f"\n[Step 2/3] News Sentiment Agent  [SKIPPED - no_news=True]")
        news_signals = {}
    else:
        print(f"\n[Step 2/3] News Sentiment Agent  ({len(tickers)} tickers)")
        news_signals = NewsAgent.score_tickers(
            tickers,
            max_headlines=max_news_headlines,
            device=device,
        )
        covered = sum(1 for v in news_signals.values() if v["news_count"] > 0)
        print(f"  -> News found for {covered}/{len(tickers)} tickers")

    # --- Agent 3: Portfolio construction ---
    print(f"\n[Step 3/3] Portfolio Constructor Agent")
    portfolio = PortfolioAgent.construct(
        gnn_signals=gnn_signals,
        news_signals=news_signals,
        alpha=alpha,
        top_k=top_k,
    )
    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed:.1f}s")

    return portfolio


def display(portfolio: pd.DataFrame, top_k: int = 10) -> None:
    """Pretty-print the portfolio recommendation table."""
    buys  = portfolio[portfolio["action"] == "BUY"]
    sells = portfolio[portfolio["action"] == "SELL"]
    holds = portfolio[portfolio["action"] == "HOLD"]

    print(f"\n{'='*72}")
    print(f"{'PORTFOLIO RECOMMENDATIONS':^72}")
    print(f"{'='*72}")

    cols = ["ticker", "gnn_rank", "sentiment_score", "news_count", "final_score", "action"]
    display_cols = [c for c in cols if c in portfolio.columns]

    print(f"\n{'--- BUY CANDIDATES (top predicted return + sentiment) ---':}")
    if not buys.empty:
        print(buys[display_cols].to_string(index=True))

    print(f"\n{'--- SELL / SHORT CANDIDATES ---':}")
    if not sells.empty:
        print(sells[display_cols].to_string(index=True))

    if not holds.empty:
        print(f"\n(+{len(holds)} HOLD stocks not shown)")
    print(f"{'='*72}")
