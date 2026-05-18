"""
Market Context Agent — fetches NIFTY 50 and India VIX context via yfinance.
Derives a market regime and a confidence multiplier used to adjust BUY scores.

Output:
    nifty_return_1d     : today's NIFTY 50 daily return
    nifty_trend_5d      : 5-day cumulative return
    nifty_ma20_ratio    : price / 20-day SMA (> 1 = above MA)
    vix_level           : current India VIX
    vix_30d_avg         : 30-day VIX average
    market_regime       : 'BULL' | 'BEAR' | 'SIDEWAYS'
    confidence_multiplier : float ∈ [0.70, 1.20]
    fetch_date          : actual date used (≤ requested date_str)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NSEI_TICKER     = "^NSEI"
INDIAVIX_TICKER = "^INDIAVIX"
VIX_HIGH = 20.0
VIX_LOW  = 14.0

_SAFE_DEFAULTS = {
    "nifty_return_1d":       0.0,
    "nifty_trend_5d":        0.0,
    "nifty_ma20_ratio":      1.0,
    "vix_level":             15.0,
    "vix_30d_avg":           15.0,
    "market_regime":         "SIDEWAYS",
    "confidence_multiplier": 1.0,
    "fetch_date":            "unknown",
}


def _fetch_index(ticker: str, end_date: str, lookback_days: int = 90) -> pd.DataFrame:
    """Fetch OHLCV ending on or before end_date, going back lookback_days."""
    try:
        import yfinance as yf
        end   = pd.to_datetime(end_date) + pd.Timedelta(days=2)   # buffer for weekends
        start = end - pd.Timedelta(days=lookback_days + 10)
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


def _classify_regime(trend_5d: float, ma20_ratio: float) -> str:
    if trend_5d > 0.005 and ma20_ratio > 1.0:
        return "BULL"
    if trend_5d < -0.005 and ma20_ratio < 1.0:
        return "BEAR"
    return "SIDEWAYS"


def _confidence_multiplier(vix: float, regime: str) -> float:
    if regime == "BULL" and vix < VIX_LOW:
        mult = 1.10
    elif regime == "BEAR" and vix > VIX_HIGH:
        mult = 0.80
    else:
        mult = 1.00
    return float(np.clip(mult, 0.70, 1.20))


class MarketContextAgent:
    """Fetch NIFTY 50 + India VIX regime context. All data via yfinance (free)."""

    def fetch(self, date_str: str) -> dict:
        """
        Fetch market context for (or just before) date_str.
        Returns safe defaults on any data failure — never raises.
        """
        print(f"[MacroAgent] Fetching NIFTY 50 + India VIX context...")
        try:
            nifty = _fetch_index(NSEI_TICKER, date_str)
            vix   = _fetch_index(INDIAVIX_TICKER, date_str)

            if nifty.empty:
                print("  [MacroAgent] NIFTY data unavailable, using defaults")
                return {**_SAFE_DEFAULTS, "fetch_date": date_str}

            target = pd.to_datetime(date_str).normalize()
            nifty.index = pd.to_datetime(nifty.index).normalize()
            nifty = nifty[nifty.index <= target]
            if nifty.empty:
                return {**_SAFE_DEFAULTS, "fetch_date": date_str}

            close = nifty["Close"].squeeze().dropna()
            fetch_date = str(close.index[-1].date())

            ret_1d   = float(close.pct_change().iloc[-1]) if len(close) >= 2 else 0.0
            trend_5d = float((close.iloc[-1] / close.iloc[-min(5, len(close))]) - 1)
            ma20     = float(close.rolling(20, min_periods=5).mean().iloc[-1])
            ma20_ratio = float(close.iloc[-1] / ma20) if ma20 > 0 else 1.0

            vix_level = 15.0
            vix_30d   = 15.0
            if not vix.empty:
                vix.index = pd.to_datetime(vix.index).normalize()
                vix = vix[vix.index <= target]
                if not vix.empty:
                    vc = vix["Close"].squeeze().dropna()
                    vix_level = float(vc.iloc[-1])
                    vix_30d   = float(vc.rolling(30, min_periods=5).mean().iloc[-1])

            regime = _classify_regime(trend_5d, ma20_ratio)
            mult   = _confidence_multiplier(vix_level, regime)

            print(
                f"  [MacroAgent] {fetch_date}: regime={regime}  "
                f"NIFTY 1d={ret_1d:+.2%}  5d={trend_5d:+.2%}  "
                f"VIX={vix_level:.1f}  multiplier={mult:.2f}"
            )
            return {
                "nifty_return_1d":       round(ret_1d, 4),
                "nifty_trend_5d":        round(trend_5d, 4),
                "nifty_ma20_ratio":      round(ma20_ratio, 4),
                "vix_level":             round(vix_level, 2),
                "vix_30d_avg":           round(vix_30d, 2),
                "market_regime":         regime,
                "confidence_multiplier": mult,
                "fetch_date":            fetch_date,
            }

        except Exception as e:
            print(f"  [MacroAgent] FAILED: {e} — using defaults")
            return {**_SAFE_DEFAULTS, "fetch_date": date_str}
