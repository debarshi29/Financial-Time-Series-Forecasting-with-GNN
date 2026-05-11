"""
GNN Prediction Agent — loads the hybrid THGNN×MaGNet checkpoint and runs inference
for a given date, returning per-ticker predicted return rankings.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

# Allow imports from the THGNN_MaGNet module
MAGNET_DIR = Path(__file__).resolve().parents[2] / "THGNN_MaGNet"
sys.path.insert(0, str(MAGNET_DIR))

from model.hybrid_model import HybridStockModel


DATA_DIR      = Path(__file__).resolve().parents[2] / "THGNN" / "data"
TRAIN_DATA_DIR = DATA_DIR / "data_train_predict"
DAILY_STOCK_DIR = DATA_DIR / "daily_stock"
MODEL_DIR     = Path(__file__).resolve().parents[2] / "THGNN_MaGNet" / "data" / "model_saved"

# Fallback: look in THGNN model_saved if THGNN_MaGNet doesn't exist yet
if not MODEL_DIR.exists():
    MODEL_DIR = DATA_DIR / "model_saved"


def _resolve_checkpoint(checkpoint_path: str | Path | None) -> Path:
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    # Search both possible model dirs
    search_dirs = [
        Path(__file__).resolve().parents[2] / "THGNN_MaGNet" / "data" / "model_saved",
        DATA_DIR / "model_saved",
    ]
    for d in search_dirs:
        candidates = sorted(d.glob("*hybrid_best.dat"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
    # Fall back to any checkpoint
    for d in search_dirs:
        candidates = sorted(d.glob("*_best.dat"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
    raise FileNotFoundError("No hybrid checkpoint found. Train the model first.")


def _find_pkl_for_date(date_str: str) -> Path:
    """Find the graph pkl file closest to (but not after) the given date."""
    target = pd.to_datetime(date_str).normalize()
    pkls = sorted(TRAIN_DATA_DIR.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No .pkl files in {TRAIN_DATA_DIR}")
    dates = [pd.to_datetime(p.stem).normalize() for p in pkls]

    # Find closest on-or-before date
    before = [(d, p) for d, p in zip(dates, pkls) if d <= target]
    if before:
        return before[-1][1]
    return pkls[0]  # fallback: earliest available


def _load_model(ckpt_path: Path, device: torch.device) -> HybridStockModel:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model = HybridStockModel(
        in_features=int(cfg.get("in_features", 12)),
        embed_dim=int(cfg.get("embed_dim", 32)),
        num_mage_layers=int(cfg.get("num_mage_layers", 1)),
        num_moe_experts=int(cfg.get("num_moe_experts", 2)),
        num_mha_heads=int(cfg.get("num_mha_heads", 2)),
        gat_heads=int(cfg.get("gat_heads", 4)),
        gat_out_features=int(cfg.get("gat_out_features", 8)),
        num_hyper_edges=int(cfg.get("num_hyper_edges", 16)),
        num_tch_hyper_edges=int(cfg.get("num_tch_hyper_edges", 16)),
        num_tch_heads=int(cfg.get("num_tch_heads", 4)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _get_tickers_for_date(date_str: str) -> list[str]:
    """Get ordered list of tickers for the date (same order as pkl features)."""
    target = pd.to_datetime(date_str).normalize()
    daily_stock_files = sorted(DAILY_STOCK_DIR.glob("*.csv"))
    if not daily_stock_files:
        return []
    dates = [pd.to_datetime(p.stem).normalize() for p in daily_stock_files]
    before = [(d, p) for d, p in zip(dates, daily_stock_files) if d <= target]
    if not before:
        csv_path = daily_stock_files[0]
    else:
        csv_path = before[-1][1]
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "code" in df.columns:
            return df["code"].tolist()
    except Exception:
        pass
    return []


class GNNAgent:
    """Stateful agent — load checkpoint once, call predict() multiple times."""

    def __init__(self, checkpoint_path: str | Path | None = None, device: str | None = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.ckpt_path = _resolve_checkpoint(checkpoint_path)
        print(f"[GNNAgent] Loading checkpoint: {self.ckpt_path.name}")
        self.model = _load_model(self.ckpt_path, self.device)
        print(f"[GNNAgent] Model ready on {self.device}")

    def predict(self, date_str: str) -> dict[str, float]:
        """Run inference for a given date. Returns {ticker: predicted_return_score}."""
        pkl_path = _find_pkl_for_date(date_str)
        print(f"[GNNAgent] Using data: {pkl_path.name} for date={date_str}")

        sample = pickle.load(open(pkl_path, "rb"))
        features = torch.tensor(sample["features"], dtype=torch.float32).to(self.device)
        pos_adj  = torch.tensor(sample["pos_adj"],  dtype=torch.float32).to(self.device)
        neg_adj  = torch.tensor(sample["neg_adj"],  dtype=torch.float32).to(self.device)
        mask     = sample.get("mask", [True] * features.shape[0])

        if isinstance(mask, torch.Tensor):
            mask_bool = mask.bool().tolist()
        else:
            mask_bool = [bool(m) for m in mask]

        with torch.no_grad():
            preds = self.model(features, pos_adj, neg_adj).squeeze(-1).cpu().numpy()

        tickers = _get_tickers_for_date(date_str)
        n = len(preds)

        if len(tickers) != n:
            tickers = [f"STOCK_{i}" for i in range(n)]

        return {
            ticker: float(preds[i])
            for i, ticker in enumerate(tickers[:n])
            if i < len(mask_bool) and mask_bool[i]
        }
