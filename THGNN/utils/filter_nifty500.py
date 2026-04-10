"""Filter Nifty 500 symbols by minimum trading history and save the valid list."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf


def filter_nifty500(
    csv_path: Path,
    start: str = "2015-01-01",
    min_days: int = 1800,
) -> List[str]:
    """Load Nifty 500 symbols from NSE CSV and filter by minimum trading history.

    Args:
        csv_path: Path to ``ind_nifty500list.csv`` downloaded from NSE.
        start: Start date for history check (YYYY-MM-DD).
        min_days: Minimum number of trading days required (default 1800 ≈ 7.5 years).

    Returns:
        List of ticker symbols (suffixed with ``.NS``) that pass the history filter.
    """
    nifty500 = pd.read_csv(csv_path)
    symbols = [s.strip() + ".NS" for s in nifty500["Symbol"]]

    valid: List[str] = []
    print(f"Checking history for {len(symbols)} Nifty 500 symbols (min_days={min_days})...", flush=True)
    for sym in symbols:
        try:
            df = yf.download(sym, start=start, progress=False, threads=False)
            if len(df) >= min_days:
                valid.append(sym)
                print(f"  [OK] {sym} ({len(df)} days)", flush=True)
            else:
                print(f"  [SKIP] {sym} ({len(df)} days < {min_days})", flush=True)
        except Exception as exc:
            print(f"  [ERROR] {sym}: {exc}", flush=True)

    print(f"\n{len(valid)} / {len(symbols)} stocks passed the history filter.", flush=True)
    return valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to ind_nifty500list.csv downloaded from NSE.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date for history check (YYYY-MM-DD). Default: 2015-01-01.",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=1800,
        help="Minimum number of trading days required. Default: 1800 (~7.5 years).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the valid ticker list as a text file (one per line).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid = filter_nifty500(args.csv_path, start=args.start, min_days=args.min_days)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(valid))
        print(f"Saved {len(valid)} tickers to {args.output}", flush=True)
    else:
        print("\nValid tickers:")
        for sym in valid:
            print(f"  {sym}")


if __name__ == "__main__":
    main()
