"""Generate stock relationship matrices for THGNN datasets."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

FEATURE_COLUMNS = ["high", "low", "close", "open", "to", "vol"]


def cal_pccs(x: np.ndarray, y: np.ndarray, n: int) -> float:
    sum_xy = np.sum(np.sum(x * y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x * x))
    sum_y2 = np.sum(np.sum(y * y))
    denominator = np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    if denominator == 0:
        return 0.0
    return float((n * sum_xy - sum_x * sum_y) / denominator)


def calculate_pccs(xs: List[np.ndarray], yss: Dict[str, List[np.ndarray]], n: int) -> np.ndarray:
    result = []
    for name in yss:
        ys = yss[name]
        tmp_res = []
        for pos, x in enumerate(xs):
            y = ys[pos]
            tmp_res.append(cal_pccs(x, y, n))
        result.append(tmp_res)
    return np.mean(result, axis=1)


def stock_cor_matrix(ref_dict: Dict[str, List[np.ndarray]], codes: List[str], n: int, processes: int = 1) -> pd.DataFrame:
    if processes > 1:
        pool = mp.Pool(processes=processes)
        args_all = [(ref_dict[code], ref_dict, n) for code in codes]
        results = [pool.apply_async(calculate_pccs, args=args) for args in args_all]
        output = [o.get() for o in results]
        data = np.stack(output)
        return pd.DataFrame(data=data, index=codes, columns=codes)

    data = np.zeros([len(codes), len(codes)])
    for i in tqdm(range(len(codes))):
        data[i, :] = calculate_pccs(ref_dict[codes[i]], ref_dict, n)
    return pd.DataFrame(data=data, index=codes, columns=codes)


def infer_relation_dates(
    dates: Iterable[pd.Timestamp],
    explicit_dates: Iterable[str] | None,
    start_date: str | None,
    end_date: str | None,
) -> List[pd.Timestamp]:
    date_index = pd.to_datetime(sorted(dates))
    if explicit_dates:
        return sorted(pd.to_datetime(explicit_dates))

    start_ts = pd.to_datetime(start_date) if start_date else date_index.min()
    end_ts = pd.to_datetime(end_date) if end_date else date_index.max()
    filtered = date_index[(date_index >= start_ts) & (date_index <= end_ts)]
    if filtered.empty:
        return []
    # Build one relation matrix per trading day to avoid month-end lookahead leakage.
    return sorted(filtered.tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data") / "nifty50.pkl",
        help="Path to the processed market data pickle file.",
    )
    parser.add_argument(
        "--relation-dir",
        type=Path,
        default=Path("data") / "relation",
        help="Directory to store the generated relation matrices.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Number of previous trading days used to compute correlations.",
    )
    parser.add_argument(
        "--dates",
        type=str,
        default=None,
        help="Comma separated list of end dates (YYYY-MM-DD) for which to build relations.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Lower bound (inclusive) when automatically inferring relation dates.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Upper bound (inclusive) when automatically inferring relation dates.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of processes to use when computing the correlation matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.data_path.open("rb") as fh:
        df = pickle.load(fh)
    df = pd.DataFrame(df)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["dt", "code"]).reset_index(drop=True)

    relation_dir = args.relation_dir
    relation_dir.mkdir(parents=True, exist_ok=True)

    all_dates = df["dt"].drop_duplicates().sort_values()
    explicit_dates = args.dates.split(",") if args.dates else None
    relation_dates = infer_relation_dates(all_dates, explicit_dates, args.start_date, args.end_date)

    if not relation_dates:
        raise RuntimeError("No relation dates could be determined. Adjust the input parameters.")

    stock_trade_data = all_dates.tolist()
    prev_date_num = args.window

    for end_data in relation_dates:
        if end_data not in stock_trade_data:
            print(f"Skipping {end_data.date()} – no trading data available.")
            continue
        end_index = stock_trade_data.index(end_data)
        if end_index < prev_date_num - 1:
            print(f"Skipping {end_data.date()} – insufficient lookback window.")
            continue
        start_data = stock_trade_data[end_index - (prev_date_num - 1)]
        df_window = df[(df["dt"] >= start_data) & (df["dt"] <= end_data)]
        codes = sorted(df_window["code"].unique().tolist())
        if len(codes) == 0:
            print(f"Skipping {end_data.date()} – no stock data available.")
            continue

        samples: Dict[str, List[np.ndarray]] = {}
        for code in codes:
            df_stock = df_window[df_window["code"] == code]
            values = df_stock[FEATURE_COLUMNS].values
            if values.T.shape[1] == prev_date_num:
                samples[code] = values.T
        if not samples:
            print(f"Skipping {end_data.date()} – insufficient aligned samples.")
            continue

        print(f"Building relation matrix ending on {end_data.date()} for {len(samples)} stocks ...")
        t1 = time.time()
        result = stock_cor_matrix(samples, list(samples.keys()), prev_date_num, processes=args.processes)
        result = result.fillna(0.0)
        np.fill_diagonal(result.values, 1.0)
        elapsed = time.time() - t1
        output_path = relation_dir / f"{end_data.strftime('%Y-%m-%d')}.csv"
        result.to_csv(output_path)
        print(f"Saved relation matrix to {output_path} (computed in {elapsed:.2f}s).")


if __name__ == "__main__":
    main()
