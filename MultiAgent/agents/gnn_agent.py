"""
GNN Prediction Agent — loads a model checkpoint (THGNN base, Hybrid, or Mamba+MoE)
and runs inference for a given date, returning per-ticker predicted return rankings.

model_variant options:
    "hybrid"  — THGNN×MaGNet (default, BiGRU+MoE+Hypergraph)
    "mamba"   — Mamba+MoE variant
    "thgnn"   — Base THGNN (GRU + heterogeneous GAT)
"""
from __future__ import annotations

import pickle
import sys
import pathlib
from pathlib import Path
from typing import Literal

import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "THGNN" / "data"
TRAIN_DATA_DIR = DATA_DIR / "data_train_predict"
DAILY_STOCK_DIR = DATA_DIR / "daily_stock"

ModelVariant = Literal["hybrid", "mamba", "thgnn"]


def _resolve_checkpoint(checkpoint_path: str | Path | None, variant: ModelVariant) -> Path:
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    search_dirs = [
        ROOT_DIR / "THGNN_MaGNet" / "data" / "model_saved",
        ROOT_DIR / "THGNN_Mamba_MoE" / "data" / "model_saved",
        DATA_DIR / "model_saved",
    ]

    patterns = {
        "hybrid": "*hybrid_best.dat",
        "mamba":  "*mamba_moe_best.dat",
        "thgnn":  "*icrank_best.dat",
    }
    pattern = patterns[variant]

    for d in search_dirs:
        candidates = sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]

    # Final fallback: any checkpoint in any search dir
    for d in search_dirs:
        candidates = sorted(d.glob("*_best.dat"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]

    raise FileNotFoundError(
        f"No checkpoint found for variant='{variant}'. Train the model first."
    )


def _find_pkl_for_date(date_str: str) -> Path:
    """Find the graph pkl file closest to (but not after) the given date."""
    target = pd.to_datetime(date_str).normalize()
    pkls = sorted(TRAIN_DATA_DIR.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No .pkl files in {TRAIN_DATA_DIR}")
    dates = [pd.to_datetime(p.stem).normalize() for p in pkls]
    before = [(d, p) for d, p in zip(dates, pkls) if d <= target]
    if before:
        return before[-1][1]
    return pkls[0]


def _load_module_from_file(module_name: str, file_path: Path):
    """Load a Python module from an explicit file path, bypassing sys.modules cache."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _patch_pathlib_for_load():
    """
    Checkpoints saved under Python 3.13 reference pathlib._local.WindowsPath.
    Python < 3.13 has no pathlib._local — create a shim so torch.load succeeds.
    """
    import types
    if "pathlib._local" not in sys.modules and not hasattr(pathlib, "_local"):
        shim = types.ModuleType("pathlib._local")
        shim.WindowsPath    = pathlib.WindowsPath
        shim.PosixPath      = pathlib.WindowsPath  # map to WindowsPath on Windows
        shim.PurePosixPath  = pathlib.PurePosixPath
        shim.PureWindowsPath = pathlib.PureWindowsPath
        sys.modules["pathlib._local"] = shim


def _load_model(ckpt_path: Path, device: torch.device, variant: ModelVariant):
    posix_path = pathlib.PosixPath
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath
        _patch_pathlib_for_load()
    try:
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
    finally:
        pathlib.PosixPath = posix_path

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    if variant == "thgnn":
        model_file = ROOT_DIR / "THGNN" / "model" / "Thgnn.py"
        thgnn_mod = _load_module_from_file("thgnn_model", model_file)
        StockHeteGAT = thgnn_mod.StockHeteGAT
        model = StockHeteGAT(
            in_features=int(cfg.get("in_features", 12)),
            out_features=int(cfg.get("out_features", 8)),
            num_heads=int(cfg.get("num_heads", 8)),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            num_layers=int(cfg.get("num_layers", 1)),
            predictor_out_dim=1,
            predictor_activation=cfg.get("predictor_activation", None),
            dropout=0.0,
        ).to(device)
    else:
        # Both "hybrid" and "mamba" use HybridStockModel from different directories.
        # Load from explicit file path to avoid Python module-cache conflicts.
        variant_dir = ROOT_DIR / ("THGNN_MaGNet" if variant == "hybrid" else "THGNN_Mamba_MoE")
        model_file  = variant_dir / "model" / "hybrid_model.py"
        mod_name    = f"hybrid_model_{variant}"
        # Add variant dir so relative imports inside hybrid_model.py resolve
        if str(variant_dir) not in sys.path:
            sys.path.insert(0, str(variant_dir))
        hybrid_mod = _load_module_from_file(mod_name, model_file)
        HybridStockModel = hybrid_mod.HybridStockModel
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

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        model_variant: ModelVariant = "hybrid",
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.variant = model_variant
        self.ckpt_path = _resolve_checkpoint(checkpoint_path, model_variant)
        print(f"[GNNAgent] variant={model_variant}  checkpoint={self.ckpt_path.name}")
        self.model = _load_model(self.ckpt_path, self.device, model_variant)
        print(f"[GNNAgent] Model ready on {self.device}")

    def predict(self, date_str: str) -> dict[str, float]:
        """Run inference for a given date. Returns {ticker: predicted_return_score}."""
        pkl_path = _find_pkl_for_date(date_str)
        print(f"[GNNAgent] Using data: {pkl_path.name} for date={date_str}")

        sample = pickle.load(open(pkl_path, "rb"))
        features = torch.as_tensor(sample["features"], dtype=torch.float32).to(self.device)
        pos_adj  = torch.as_tensor(sample["pos_adj"],  dtype=torch.float32).to(self.device)
        neg_adj  = torch.as_tensor(sample["neg_adj"],  dtype=torch.float32).to(self.device)
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
