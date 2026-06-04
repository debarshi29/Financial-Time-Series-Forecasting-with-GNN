# THGNN Mamba-MoE — Experimental Mamba-SSM Temporal Encoder

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Experimental-orange)](.)

An experimental variant of the hybrid THGNN × MaGNet architecture that replaces the **BiGRU temporal mixer** with a **Mamba State Space Model (SSM)** block. All other components — the MoE expert routing layer, dual hypergraph paths (TCH + GPH), semantic attention fusion, and composite IC-ranked loss — are kept identical to `THGNN_MaGNet`.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full design documentation.

---

## Motivation

Mamba [Gu & Dao, 2024] is a selective SSM that offers **linear-complexity sequence modelling** without the quadratic cost of self-attention. It uses input-dependent state-space parameters to selectively filter information along the sequence dimension, providing an inductive bias well-suited to long financial time series where most timesteps carry low signal. This experiment tests whether Mamba's selective filtering improves upon the BiGRU baseline in the temporal encoder role.

**Hypothesis:** Mamba's selective state transitions will more effectively suppress noise in OHLCV sequences, yielding better Spearman IC on NIFTY 500 at equivalent parameter count.

---

## Architecture Diff vs. THGNN_MaGNet

```
THGNN_MaGNet Temporal Encoder:       THGNN_Mamba_MoE Temporal Encoder:
─────────────────────────────         ────────────────────────────────────
  BiGRU  (fwd + bwd context)    →       Mamba SSM  (selective state space)
  Sparse MoE  (expert routing)          Sparse MoE  (expert routing)  [same]
  Multi-Head Attention                  Multi-Head Attention           [same]
```

All downstream components (TCH path, co-movement GAT + GPH path, semantic fusion, PairNorm-SI, predictor head) are identical to `THGNN_MaGNet`.

---

## Results

| Metric | THGNN Baseline | THGNN × MaGNet | THGNN Mamba-MoE |
|---|:---:|:---:|:---:|
| Spearman IC (mean) | 0.028 | **0.041** | 0.035 |
| Directional Accuracy | 51.3% | **53.2%** | 52.5% |

Mamba-MoE outperforms the baseline but falls short of the BiGRU-based hybrid. The BiGRU's bidirectional context appears to provide a stronger inductive bias for short financial windows (T = 20 days) than Mamba's selective filtering. Mamba may be more competitive over longer lookback windows.

---

## Quick Start

```bash
pip install uv
cd THGNN_Mamba_MoE
uv sync
```

### Train

```bash
python train_hybrid.py \
  --embed-dim 64 \
  --train-start-date 2015-01-01 --train-end-date 2023-12-31 \
  --test-start-date  2024-01-01 --test-end-date  2026-02-28 \
  --epochs 80 --lr 1e-4 --patience 15
```

Artifacts are saved with `_mamba_moe_*` suffixes under `../THGNN/data/` — they do not overwrite `_hybrid_*` checkpoints from the BiGRU run.

### Backtest

```bash
python backtest_hybrid.py \
  --checkpoint ../THGNN/data/model_saved/2023-12-29_mamba_moe_best.dat \
  --start-date 2024-01-01 --end-date 2026-02-28 --top-k 5
```

---

## File Reference

| File | Purpose |
|---|---|
| `model/hybrid_model.py` | Mamba-based temporal encoder + full hybrid architecture |
| `train_hybrid.py` | Training script (identical interface to THGNN_MaGNet) |
| `backtest_hybrid.py` | Backtest evaluation (saves with `_mamba_moe_*` names) |
| `data_loader.py` | Shared data pipeline (reads THGNN graph `.pkl` files) |
| `plot_live_predictions.py` | Live prediction charts |
| `ARCHITECTURE.md` | Mamba-specific design notes and comparison with BiGRU variant |

---

## Notes

- Mamba requires `mamba-ssm` which has strict CUDA version requirements. Ensure your environment matches CUDA 12.4 before running `uv sync`.
- If Mamba installation fails, fall back to `THGNN_MaGNet` which uses standard BiGRU and has no exotic dependencies.
- Checkpoint files use the `_mamba_moe_best.dat` naming convention to avoid collision with hybrid checkpoints.
