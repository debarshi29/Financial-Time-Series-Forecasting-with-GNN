"""Utility to download and preprocess Nifty 50 market data for THGNN."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import numpy as np
import yfinance as yf

NIFTY50_TICKERS: List[str] = [
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BPCL.NS",
    "BHARTIARTL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DIVISLAB.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "ITC.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LTIM.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NTPC.NS",
    "NESTLEIND.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SHRIRAMFIN.NS",
    "SBIN.NS",
    "SUNPHARMA.NS",
    "TCS.NS",
    "TATACONSUM.NS",
    "TRENT.NS",
    "TATASTEEL.NS",
    "TECHM.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS",
]
NIFTY50_TICKERS_TRIAL: List[str] = ["ADANIENT.NS", "TCS.NS"]
    


FEATURE_COLUMNS = ["open", "high", "low", "close", "to", "vol"]


def _to_iterable(value: str | Iterable[str] | None) -> Iterable[str]:
    if value is None:
        return NIFTY50_TICKERS
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def _normalize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    if data.index.tz is not None:
        data = data.tz_localize(None)
    data = data.sort_index()

    if "Adj Close" in data.columns:
        # 1. Calculate the base ratio
        ratio = data["Adj Close"] / data["Close"]

        # 2. Clean infinities (from division by 0) and NaNs (from 0/0)
        # We replace inf and -inf with np.nan so we can handle all "bad" values at once
        ratio_clean = ratio.replace([np.inf, -np.inf], np.nan)

        # 3. Create the ratio for Volume
        #    We replace 0s AND NaNs with 1.0 (so we never divide by zero or NaN)
        volume_ratio = ratio_clean.replace(0, 1.0).fillna(1.0)
        
        # 4. Create the ratio for OHLC
        #    We only replace 0 with 1.0. We keep NaNs, as they will be dropped later.
        ratio = ratio_clean.replace(0, 1.0)

        # 5. Apply the ratios
        data["Open"] = data["Open"] * ratio
        data["High"] = data["High"] * ratio
        data["Low"] = data["Low"] * ratio
        data["Close"] = data["Adj Close"]   # final Close is adjusted close
        data["Volume"] = data["Volume"] / volume_ratio  # adjust volume inversely

        # Drop the old Adj Close column
        data = data.drop(columns=["Adj Close"])

    return data


def _build_features(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["dt", "code", *FEATURE_COLUMNS, "label"])

    working = data.copy()
    working["Turnover"] = working["Close"] * working["Volume"]

    feat = pd.DataFrame(index=working.index)
    feat["open"] = data["Open"].pct_change()
    feat["high"] = data["High"].pct_change()
    feat["low"] = data["Low"].pct_change()
    feat["close"] = data["Close"].pct_change()
    feat["to"] = working["Turnover"].replace(0, np.nan).pct_change()
    feat["vol"] = data["Volume"].replace(0, np.nan).pct_change()

    feat["label"] = feat["close"].shift(-1)
    feat = feat.dropna().reset_index().rename(columns={"Date": "dt"})
    if feat.empty:
        return feat

    feat["dt"] = feat["dt"].dt.strftime("%Y-%m-%d")
    feat["code"] = ticker
    columns = ["dt", "code", *FEATURE_COLUMNS, "label"]
    return feat[columns]


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def download_market_data(
    tickers: Iterable[str],
    start: str,
    end: str | None,
    output: Path,
    *,
    pause: float = 1.0,
    max_retries: int = 3,
    csv_dir: Path | None = None,
) -> pd.DataFrame:
    output.parent.mkdir(parents=True, exist_ok=True)
    if csv_dir is not None:
        csv_dir.mkdir(parents=True, exist_ok=True)

    tickers = list(tickers)
    print(f"Preparing to download {len(tickers)} tickers...", flush=True)

    frames: List[pd.DataFrame] = []
    for ticker in tickers:
        print(f"Downloading {ticker} ...", flush=True)
        attempt = 0
        raw = pd.DataFrame()
        normalized = pd.DataFrame()
        while attempt < max_retries:
            attempt += 1
            try:
                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                # ticker_obj = yf.Ticker(ticker)
                # raw = ticker_obj.history(
                #     start=start,
                #     end=end,
                #     auto_adjust=False,
                # )                
                if raw.empty:
                    raise RuntimeError("empty response")
                normalized = _normalize_raw(raw)
                if normalized.empty:
                    raise RuntimeError("normalized dataset is empty")
            except Exception as exc:  # pragma: no cover - network errors are runtime specific
                print(f"  Attempt {attempt} failed for {ticker}: {exc}", flush=True)
                time.sleep(pause)
                continue
            break

        if normalized.empty:
            print(f"  Giving up on {ticker} after {max_retries} attempts.", flush=True)
            continue

        if csv_dir is not None:
            enriched = normalized.copy()
            enriched["RSI"] = _compute_rsi(enriched["Close"])
            macd, macd_signal, macd_hist = _compute_macd(enriched["Close"])
            enriched["MACD"] = macd
            enriched["MACD_Signal"] = macd_signal
            enriched["MACD_Hist"] = macd_hist
            enriched = enriched.dropna().reset_index().rename(columns={"index": "Date"})
            csv_path = csv_dir / f"{ticker.replace('.', '_')}.csv"
            enriched.to_csv(csv_path, index=False)
            print(f"  Saved adjusted OHLCV with RSI & MACD to {csv_path}", flush=True)

        frame = _build_features(normalized, ticker)
        if frame.empty:
            print(f"Warning: no usable data for {ticker} in the given period.", flush=True)
            continue
        frames.append(frame)

    if not frames:
        raise RuntimeError("No market data could be downloaded. Please check ticker symbols and date range.")

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sort_values(["dt", "code"]).reset_index(drop=True)
    dataset.to_pickle(output)
    print(f"Saved dataset with {len(dataset)} rows to {output}.", flush=True)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma separated list of ticker symbols. Defaults to the Nifty 50 constituents.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date for the historical data in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end date for the historical data in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "nifty50.pkl",
        help="Path where the processed dataset will be stored (pickle format).",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Optional directory to also export per-ticker adjusted OHLCV with RSI & MACD.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Seconds to wait between retry attempts when downloading data.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of download attempts per ticker before skipping it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = list(_to_iterable(args.tickers))
    download_market_data(
        tickers=tickers,
        start=args.start,
        end=args.end,
        output=args.output,
        pause=args.pause,
        max_retries=args.max_retries,
        csv_dir=args.csv_dir,
    )


if __name__ == "__main__":
    main()
