"""
Rebuild data/data_train_predict/*.pkl with a configurable adjacency threshold.

The original data used corr_threshold=0.1 → ~49% edge density (near-fully connected).
Raising to 0.3 gives ~15% density, giving the GAT real structure to exploit.

Usage:
    python rebuild_graph_data.py                  # default threshold=0.3
    python rebuild_graph_data.py --threshold 0.5  # sparser graph
    python rebuild_graph_data.py --threshold 0.1  # restore original
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
NIFTY_PKL = DATA_DIR / "nifty50.pkl"
RELATION_DIR = DATA_DIR / "relation"
OUT_DIR = DATA_DIR / "data_train_predict"

FEATURE_COLS = ["open", "high", "low", "close", "to", "vol"]
LABEL_COL = "label"
WINDOW = 20       # lookback days for features
HORIZONS = 3      # number of forward-return labels (t+1, t+2, t+3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Minimum absolute correlation for a positive edge (default 0.3). "
                        "Negative edge threshold is the same value with opposite sign.")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def load_nifty(path: Path) -> pd.DataFrame:
    df = pickle.load(open(path, "rb"))
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["dt", "code"]).reset_index(drop=True)
    return df


def build_sample(
    date: pd.Timestamp,
    stocks: list[str],
    df: pd.DataFrame,
    corr_matrix: np.ndarray,
    threshold: float,
) -> dict | None:
    """Build one sample dict for the given date."""
    date_idx_map = {
        code: df[(df["code"] == code) & (df["dt"] == date)].index
        for code in stocks
    }

    # Require all stocks to have data on this date
    for code, idx in date_idx_map.items():
        if len(idx) == 0:
            return None

    features_list = []
    labels_list = []
    mask_list = []

    for code in stocks:
        row_idx = date_idx_map[code][0]
        stock_df = df[df["code"] == code].sort_values("dt").reset_index(drop=True)
        pos_in_stock = stock_df.index[stock_df["dt"] == date].tolist()
        if not pos_in_stock:
            return None
        t = pos_in_stock[0]

        # Need WINDOW days before (inclusive of today) for features
        if t < WINDOW:
            return None
        feat_rows = stock_df.iloc[t - WINDOW + 1: t + 1]
        if len(feat_rows) != WINDOW:
            return None

        # Need HORIZONS days after for labels
        if t + HORIZONS >= len(stock_df):
            return None
        label_rows = stock_df.iloc[t + 1: t + 1 + HORIZONS]
        if len(label_rows) != HORIZONS:
            return None

        features_list.append(feat_rows[FEATURE_COLS].values.astype(np.float32))
        labels_list.append(label_rows[LABEL_COL].values.astype(np.float32))
        mask_list.append(True)

    if len(features_list) != len(stocks):
        return None

    features = torch.tensor(np.stack(features_list), dtype=torch.float32)  # (N, 20, 6)
    labels = torch.tensor(np.stack(labels_list), dtype=torch.float32)       # (N, 3)

    # Build adjacency from correlation matrix
    pos_adj = (corr_matrix > threshold).astype(np.float32)
    neg_adj = (corr_matrix < -threshold).astype(np.float32)
    np.fill_diagonal(pos_adj, 0)
    np.fill_diagonal(neg_adj, 0)

    return {
        "features": features,
        "labels": labels,
        "pos_adj": torch.tensor(pos_adj, dtype=torch.float32),
        "neg_adj": torch.tensor(neg_adj, dtype=torch.float32),
        "mask": mask_list,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {NIFTY_PKL}...")
    df = load_nifty(NIFTY_PKL)

    relation_files = sorted(RELATION_DIR.glob("*.csv"))
    if not relation_files:
        raise RuntimeError(f"No relation CSV files found in {RELATION_DIR}")

    print(f"Found {len(relation_files)} relation files.")
    print(f"Adjacency threshold: ±{args.threshold}")

    # Pre-compute the set of existing dates to avoid rebuilding if not needed
    existing = {p.stem for p in args.out_dir.glob("*.pkl")}
    print(f"Existing pkl files in output dir: {len(existing)}")

    built = 0
    skipped = 0

    for rel_file in tqdm(relation_files, desc="Building samples"):
        date_str = rel_file.stem          # e.g. "2020-08-21"
        date = pd.Timestamp(date_str)

        # Load correlation matrix
        corr_df = pd.read_csv(rel_file, index_col=0).astype(float)
        stocks = list(corr_df.index)
        corr_matrix = corr_df.values.copy()
        np.fill_diagonal(corr_matrix, 0)

        sample = build_sample(date, stocks, df, corr_matrix, args.threshold)
        if sample is None:
            skipped += 1
            continue

        out_path = args.out_dir / f"{date_str}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(sample, f)
        built += 1

    print(f"\nDone. Built: {built}, Skipped (insufficient history/future): {skipped}")
    print(f"Output: {args.out_dir}")

    # Report density stats on a few samples
    sample_files = sorted(args.out_dir.glob("*.pkl"))[::200]
    densities = []
    for fp in sample_files:
        s = pickle.load(open(fp, "rb"))
        densities.append(float(s["pos_adj"].mean()))
    print(f"\nPos-adj density (sampled): mean={np.mean(densities):.3f}, "
          f"min={np.min(densities):.3f}, max={np.max(densities):.3f}")


if __name__ == "__main__":
    main()
