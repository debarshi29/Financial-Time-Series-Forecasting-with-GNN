"""
Risk Agent — computes per-ticker risk metrics for BUY and SELL candidates.
All data from yfinance (free, no API key required).

Output per ticker:
    beta              : market beta vs NIFTY 50
    vol_20d           : 20-day realised daily std
    pct_from_52w_high : e.g. -0.12 means 12% below the 52-week high
    pct_from_52w_low  : e.g. +0.35 means 35% above the 52-week low
    range_position    : (price - 52w_low) / (52w_high - 52w_low)  ∈ [0, 1]
    atr               : 14-day Average True Range
    current_price     : latest close price
    risk_label        : 'HIGH' | 'MEDIUM' | 'LOW' | 'UNKNOWN'
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

NSEI_TICKER = "^NSEI"
VOL_HIGH    = 0.030    # daily std > 3 % → HIGH
VOL_MEDIUM  = 0.015    # daily std > 1.5 % → MEDIUM
BETA_HIGH   = 1.5
BETA_MEDIUM = 1.0


def _fetch_history(ticker: str, end_date: str | None = None) -> pd.DataFrame:
    """Fetch ~1 year of daily OHLCV ending on or before end_date."""
    try:
        import yfinance as yf
        if end_date:
            end   = pd.to_datetime(end_date) + pd.Timedelta(days=2)
            start = end - pd.Timedelta(days=380)
        else:
            end   = pd.Timestamp.today() + pd.Timedelta(days=1)
            start = end - pd.Timedelta(days=380)
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _compute_beta(stock_returns: pd.Series, index_returns: pd.Series) -> float:
    """OLS beta. Returns 1.0 if insufficient data."""
    try:
        aligned = pd.concat([stock_returns, index_returns], axis=1).dropna()
        if len(aligned) < 20:
            return 1.0
        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 1.0
    except Exception:
        return 1.0


def _compute_atr(ohlcv: pd.DataFrame, window: int = 14) -> float:
    """14-day Average True Range. Returns NaN on failure."""
    try:
        high = ohlcv["High"].squeeze()
        low  = ohlcv["Low"].squeeze()
        close = ohlcv["Close"].squeeze()
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(window).mean().iloc[-1])
    except Exception:
        return float("nan")


def _risk_label(vol_20d: float, beta: float) -> str:
    if vol_20d > VOL_HIGH or beta > BETA_HIGH:
        return "HIGH"
    if vol_20d > VOL_MEDIUM or beta > BETA_MEDIUM:
        return "MEDIUM"
    return "LOW"


class RiskAgent:
    """Assess risk metrics for a list of tickers using yfinance (free)."""

    def __init__(self, delay_s: float = 0.5) -> None:
        self.delay_s = delay_s

    def assess(
        self,
        tickers: list[str],
        index_ticker: str = NSEI_TICKER,
        end_date: str | None = None,
    ) -> dict[str, dict]:
        """
        Parameters
        ----------
        tickers      : BUY + SELL candidate tickers (NSE .NS format already)
        index_ticker : Benchmark for beta (default ^NSEI)
        end_date     : Fetch data up to this date (YYYY-MM-DD); defaults to today

        Returns
        -------
        {ticker: {beta, vol_20d, pct_from_52w_high, pct_from_52w_low,
                  range_position, atr, current_price, risk_label}}
        """
        print(f"[RiskAgent] Fetching NIFTY 50 index history...")
        index_df = _fetch_history(index_ticker, end_date)
        if not index_df.empty:
            idx_close = index_df["Close"].squeeze()
            idx_returns = idx_close.pct_change().dropna()
        else:
            idx_returns = pd.Series(dtype=float)

        results: dict[str, dict] = {}

        for i, ticker in enumerate(tickers):
            print(f"  [RiskAgent] {i+1}/{len(tickers)} {ticker}", end=" ", flush=True)
            try:
                df = _fetch_history(ticker, end_date)
                if df.empty:
                    raise ValueError("empty history")

                close = df["Close"].squeeze()
                price = float(close.iloc[-1])
                hi52  = float(close.rolling(252, min_periods=20).max().iloc[-1])
                lo52  = float(close.rolling(252, min_periods=20).min().iloc[-1])
                stock_returns = close.pct_change().dropna()

                vol_20d = float(stock_returns.rolling(20).std().iloc[-1])
                beta    = _compute_beta(stock_returns, idx_returns) if not idx_returns.empty else 1.0
                atr     = _compute_atr(df)
                rng     = (hi52 - lo52) if (hi52 - lo52) > 0 else float("nan")

                results[ticker] = {
                    "beta":              round(beta, 3),
                    "vol_20d":           round(vol_20d, 4),
                    "pct_from_52w_high": round((price - hi52) / hi52, 4) if hi52 else float("nan"),
                    "pct_from_52w_low":  round((price - lo52) / lo52, 4) if lo52 else float("nan"),
                    "range_position":    round((price - lo52) / rng, 4) if not np.isnan(rng) else float("nan"),
                    "atr":               round(atr, 4),
                    "current_price":     round(price, 2),
                    "risk_label":        _risk_label(vol_20d, beta),
                }
                print(f"-> beta={beta:.2f}  vol={vol_20d:.3f}  {results[ticker]['risk_label']}")
            except Exception as e:
                print(f"-> FAILED ({e})")
                results[ticker] = {
                    "beta": float("nan"), "vol_20d": float("nan"),
                    "pct_from_52w_high": float("nan"), "pct_from_52w_low": float("nan"),
                    "range_position": float("nan"), "atr": float("nan"),
                    "current_price": float("nan"), "risk_label": "UNKNOWN",
                }

            if self.delay_s > 0 and i < len(tickers) - 1:
                time.sleep(self.delay_s)

        return results
