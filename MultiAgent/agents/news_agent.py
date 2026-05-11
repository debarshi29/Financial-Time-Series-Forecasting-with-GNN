"""
News Sentiment Agent — fetches recent news via yfinance, scores with FinBERT.

Output per ticker:
    sentiment_score  = mean(positive) - mean(negative)  ∈ [-1, 1]
    news_count       = number of headlines scored
    headlines        = list of (title, positive, negative, neutral) dicts
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.finbert_loader import score_texts
from utils.news_cache import get_cached, store


def _fetch_news_yfinance(ticker: str, max_items: int = 20) -> list[str]:
    """Fetch recent news headlines for a NSE ticker via yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        news = t.news or []
        titles = []
        for item in news[:max_items]:
            content = item.get("content", {})
            title = (
                content.get("title")
                or item.get("title")
                or ""
            )
            if title:
                titles.append(title)
        return titles
    except Exception as e:
        print(f"  [NewsAgent] yfinance error for {ticker}: {e}")
        return []


def score_ticker(
    ticker: str,
    max_headlines: int = 20,
    cache_max_age_s: float = 3600.0,
    device: str | None = None,
) -> dict:
    """Fetch + score news for one ticker. Returns sentiment dict."""
    # Try cache first
    cached = get_cached(ticker, max_age_s=cache_max_age_s)
    if cached is not None:
        titles = [item["title"] for item in cached]
        scored = cached  # already has scores
    else:
        titles = _fetch_news_yfinance(ticker, max_headlines)
        if not titles:
            return {
                "ticker": ticker,
                "sentiment_score": 0.0,
                "news_count": 0,
                "headlines": [],
            }
        scores = score_texts(titles, device=device)
        scored = [
            {"title": t, "positive": s["positive"], "negative": s["negative"], "neutral": s["neutral"]}
            for t, s in zip(titles, scores)
        ]
        store(ticker, scored)

    if not scored:
        return {"ticker": ticker, "sentiment_score": 0.0, "news_count": 0, "headlines": []}

    mean_pos = sum(s["positive"] for s in scored) / len(scored)
    mean_neg = sum(s["negative"] for s in scored) / len(scored)
    sentiment_score = float(mean_pos - mean_neg)

    return {
        "ticker": ticker,
        "sentiment_score": sentiment_score,
        "news_count": len(scored),
        "headlines": scored,
    }


def score_tickers(
    tickers: list[str],
    max_headlines: int = 20,
    cache_max_age_s: float = 3600.0,
    device: str | None = None,
    delay_s: float = 0.3,
) -> dict[str, dict]:
    """Score a list of tickers. Returns {ticker: sentiment_dict}."""
    results = {}
    for i, ticker in enumerate(tickers):
        print(f"  [NewsAgent] {i+1}/{len(tickers)} {ticker}", end=" ", flush=True)
        result = score_ticker(ticker, max_headlines, cache_max_age_s, device)
        results[ticker] = result
        print(f"→ score={result['sentiment_score']:+.3f}  n={result['news_count']}")
        if delay_s > 0 and i < len(tickers) - 1:
            time.sleep(delay_s)
    return results
