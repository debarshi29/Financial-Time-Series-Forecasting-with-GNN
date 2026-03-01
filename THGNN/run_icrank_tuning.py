from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "data" / "model_saved"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3-way IC-ranked tuning ablation.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--train-start-date", type=str, default="2015-01-01")
    parser.add_argument("--train-end-date", type=str, default="2023-12-29")
    parser.add_argument("--val-start-date", type=str, default="2024-01-01")
    parser.add_argument("--val-end-date", type=str, default="2025-12-31")
    return parser.parse_args()


def load_checkpoint_metrics(path: Path) -> tuple[float, float, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    val_ic = float(ckpt.get("val_ic", 0.0))
    val_mse = float(ckpt.get("val_mse", 0.0))
    epoch = int(ckpt.get("epoch", -1))
    return val_ic, val_mse, epoch


def run_one(config_name: str, base_args: argparse.Namespace, ic_weight: float, disp_weight: float, min_disp: float) -> dict:
    cmd = [
        sys.executable,
        "train_ic_ranked.py",
        "--epochs",
        str(base_args.epochs),
        "--patience",
        str(base_args.patience),
        "--train-start-date",
        base_args.train_start_date,
        "--train-end-date",
        base_args.train_end_date,
        "--val-start-date",
        base_args.val_start_date,
        "--val-end-date",
        base_args.val_end_date,
        "--ic-weight",
        str(ic_weight),
        "--dispersion-weight",
        str(disp_weight),
        "--min-dispersion-ratio",
        str(min_disp),
    ]

    print(f"\n=== Running {config_name} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=BASE_DIR)

    pre_data = base_args.train_end_date
    src = MODEL_DIR / f"{pre_data}_icrank_best.dat"
    if not src.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {src}")

    dst = MODEL_DIR / f"{pre_data}_icrank_{config_name}.dat"
    shutil.copy2(src, dst)
    val_ic, val_mse, epoch = load_checkpoint_metrics(dst)

    return {
        "name": config_name,
        "ic_weight": ic_weight,
        "disp_weight": disp_weight,
        "min_disp": min_disp,
        "best_epoch": epoch,
        "val_ic": val_ic,
        "val_mse": val_mse,
        "checkpoint": str(dst),
    }


def main() -> None:
    args = parse_args()

    configs = [
        # Safer calibration, usually better starting point.
        ("balanced", 0.25, 0.10, 0.12),
        # Your current aggressive setting.
        ("aggressive", 0.35, 0.20, 0.20),
        # Conservative regularization.
        ("conservative", 0.15, 0.05, 0.10),
    ]

    results = []
    for name, ic_w, disp_w, min_disp in configs:
        results.append(run_one(name, args, ic_w, disp_w, min_disp))

    results = sorted(results, key=lambda x: (x["val_ic"], -x["val_mse"]), reverse=True)

    print("\n=== Tuning Summary (sorted by val_ic desc, val_mse asc) ===")
    for r in results:
        print(
            f"{r['name']:12s} | "
            f"val_ic={r['val_ic']:.4f} | val_mse={r['val_mse']:.6f} | "
            f"epoch={r['best_epoch']:3d} | "
            f"ic_w={r['ic_weight']:.2f}, disp_w={r['disp_weight']:.2f}, min_disp={r['min_disp']:.2f}"
        )
        print(f"  checkpoint: {r['checkpoint']}")

    print("\nBest config:", results[0]["name"])


if __name__ == "__main__":
    main()
