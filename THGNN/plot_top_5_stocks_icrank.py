from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_loader import AllGraphDataSampler
from model.Thgnn import StockHeteGAT
from trainer.trainer import extract_data


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "data_train_predict"
DAILY_STOCK_DIR = DATA_DIR / "daily_stock"
MODEL_DIR = DATA_DIR / "model_saved"
PLOT_DIR = DATA_DIR / "plots"
NIFTY_CSV_DIR = DATA_DIR / "nifty50_csv"

PLOT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot top-5 stocks using IC-ranked checkpoint.")
    parser.add_argument("--val-start-date", type=str, default="2024-01-01")
    parser.add_argument("--val-end-date", type=str, default="2025-12-31")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path override.")
    return parser.parse_args()


def _index_for_date(file_dates: list[pd.Timestamp], date_str: str) -> int:
    dt = pd.to_datetime(date_str).normalize()
    for idx, d in enumerate(file_dates):
        if d >= dt:
            return idx
    return len(file_dates) - 1


def _resolve_checkpoint(pre_data_value: str, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        path = Path(checkpoint_arg)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    exact = MODEL_DIR / f"{pre_data_value}_icrank_best.dat"
    if exact.exists():
        return exact

    candidates = sorted(MODEL_DIR.glob("*_icrank_best.dat"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No *_icrank_best.dat found in {MODEL_DIR}")
    return candidates[-1]


def _load_checkpoint(path: Path, device: torch.device):
    # PyTorch 2.6 defaults torch.load(..., weights_only=True), which cannot
    # deserialize our full training checkpoint dict (contains pathlib objects).
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not accept weights_only argument.
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_files = sorted([p.name for p in TRAIN_DATA_DIR.glob("*.pkl")])
    if not data_files:
        raise RuntimeError(f"No training data found in {TRAIN_DATA_DIR}")
    file_dates = [pd.to_datetime(Path(name).stem).normalize() for name in data_files]

    val_start_idx = _index_for_date(file_dates, args.val_start_date)
    val_end_idx = _index_for_date(file_dates, args.val_end_date) + 1
    val_end_idx = min(val_end_idx, len(data_files))
    if val_end_idx <= val_start_idx:
        raise ValueError("Validation range is empty.")

    pre_data_value = Path(data_files[val_start_idx - 1 if val_start_idx > 0 else 0]).stem
    print(f"Validation Range: {file_dates[val_start_idx].date()} to {file_dates[val_end_idx - 1].date()}")
    print(f"Pre-data value: {pre_data_value}")

    checkpoint_path = _resolve_checkpoint(pre_data_value, args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = _load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    val_dataset = AllGraphDataSampler(
        str(TRAIN_DATA_DIR),
        mode="val",
        data_start=0,
        data_middle=val_start_idx,
        data_end=val_end_idx,
    )
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty.")

    _, _, _, sample_labels, _ = extract_data(val_dataset[0], str(device))
    target_dim = int(sample_labels.shape[-1]) if sample_labels.dim() > 1 else 1

    model = StockHeteGAT(
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_heads=int(cfg.get("num_heads", 8)),
        num_layers=int(cfg.get("num_layers", 1)),
        out_features=int(cfg.get("out_features", 32)),
        predictor_out_dim=target_dim,
        predictor_activation=None,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    results = []
    print("Running predictions...")
    for i in tqdm(range(len(val_dataset)), unit="day"):
        pkl_name = val_dataset.gnames_all[i]
        date_str = Path(pkl_name).stem
        daily_file = DAILY_STOCK_DIR / f"{date_str}.csv"
        if not daily_file.exists():
            continue
        df = pd.read_csv(daily_file, dtype=object)
        sample = val_dataset[i]
        pos_adj, neg_adj, features, labels, mask = extract_data(sample, str(device))
        with torch.no_grad():
            logits = model(features, pos_adj, neg_adj)
        preds = logits.detach().cpu().numpy()
        lbls = labels.detach().cpu().numpy()
        if preds.ndim == 1:
            preds = preds[:, None]
        if lbls.ndim == 1:
            lbls = lbls[:, None]

        for idx, row in df.iterrows():
            if idx >= len(preds):
                break
            results.append(
                {
                    "dt": pd.to_datetime(row["dt"]),
                    "code": row["code"],
                    "actual_return": float(lbls[idx, 0]),
                    "predicted_return": float(preds[idx, 0]),
                }
            )

    df_results = pd.DataFrame(results)
    if df_results.empty:
        raise RuntimeError("No predictions generated.")

    metrics = []
    for code, group in df_results.groupby("code"):
        if len(group) < 10:
            continue
        mse = np.mean((group["actual_return"] - group["predicted_return"]) ** 2)
        metrics.append({"code": code, "mse": mse})
    df_metrics = pd.DataFrame(metrics)
    if df_metrics.empty:
        raise RuntimeError("No valid stock metrics computed.")

    top_5 = df_metrics.sort_values("mse").head(5)
    print("\nTop 5 Best Stocks:")
    print(top_5)

    for _, row in top_5.iterrows():
        code = row["code"]
        mse = float(row["mse"])
        stock_data = df_results[df_results["code"] == code].sort_values("dt")
        pred_ret = stock_data["predicted_return"].values
        act_ret = stock_data["actual_return"].values
        ic = np.corrcoef(pred_ret, act_ret)[0, 1] if len(stock_data) > 1 else np.nan
        sign_acc = np.mean(np.sign(pred_ret) == np.sign(act_ret))
        pred_std = float(np.std(pred_ret))
        act_std = float(np.std(act_ret))
        spread_ratio = pred_std / act_std if act_std > 0 else np.nan

        csv_path = NIFTY_CSV_DIR / f"{code.replace('.', '_')}.csv"
        if not csv_path.exists():
            print(f"Skipping {code}: price csv not found ({csv_path})")
            continue
        df_price = pd.read_csv(csv_path)
        df_price["Date"] = pd.to_datetime(df_price["Date"])
        merged = pd.merge(stock_data, df_price, left_on="dt", right_on="Date", how="inner")
        if merged.empty:
            print(f"Skipping {code}: no overlapping dates.")
            continue

        merged["actual_next_price"] = merged["Close"] * (1 + merged["actual_return"])
        merged["predicted_next_price"] = merged["Close"] * (1 + merged["predicted_return"])

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(merged["dt"], merged["actual_return"], color="black", alpha=0.5, label="Actual Return (t+1)")
        axes[0].plot(merged["dt"], merged["predicted_return"], color="blue", alpha=0.9, label="Predicted Return (t+1)")
        axes[0].set_title(f"{code} - Next Day Returns (MSE: {mse:.6f})")
        axes[0].set_ylabel("Return")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].text(
            0.01,
            0.98,
            f"IC: {ic:.4f}\nSign Acc: {sign_acc:.2%}\nPred Std: {pred_std:.6f}\nAct Std: {act_std:.6f}\nSpread Ratio: {spread_ratio:.3f}",
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        axes[1].plot(merged["dt"], merged["actual_next_price"], color="green", linewidth=2, label="Actual Next Price (P_t+1)")
        axes[1].plot(
            merged["dt"],
            merged["predicted_next_price"],
            color="red",
            linestyle="--",
            alpha=0.8,
            label="Predicted Next Price (Pred_P_t+1)",
        )
        axes[1].plot(merged["dt"], merged["Close"], color="gray", linestyle=":", alpha=0.5, label="Current Price (P_t)")
        axes[1].set_title(f"{code} - Next Day Price Prediction")
        axes[1].set_ylabel("Price")
        axes[1].set_xlabel("Date")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = PLOT_DIR / f"top5_icrank_{code}.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    main()
