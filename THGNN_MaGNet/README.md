# THGNN × MaGNet — Hybrid Temporal-Hypergraph Stock Forecasting

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

This module implements the **central algorithmic contribution** of the thesis: a hybrid architecture that extends the baseline THGNN with a MaGNet-style temporal encoder, a Temporal-Causal Hypergraph (TCH) relational path, and a Global Probabilistic Hypergraph (GPH) co-movement path, fused via semantic attention.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full design documentation.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Architecture](#2-architecture)
3. [Loss Function](#3-loss-function)
4. [Quick Start](#4-quick-start)
5. [Hyperparameter Reference](#5-hyperparameter-reference)
6. [Checkpoints](#6-checkpoints)
7. [File Reference](#7-file-reference)

---

## 1. Motivation

The baseline THGNN encodes temporal dynamics with a single GRU and captures relational structure through heterogeneous GAT over positive/negative correlation edges. Two limitations motivate this extension:

- **Temporal encoder capacity.** A single-layer GRU encodes each stock independently. It cannot capture bidirectional temporal context, lacks the inductive bias to route different stocks to specialised processing paths, and has limited cross-timestep attention.

- **Relational expressivity.** Binary pairwise adjacency from rolling Pearson correlation ignores multi-stock thematic groupings (sector, market-cap, macro sensitivity) and asynchronous lead-lag relationships between stocks.

This hybrid architecture addresses both:

1. The **MaGNet Temporal Encoder** replaces the GRU with a BiGRU for bidirectional context, a Sparse Mixture-of-Experts layer for per-stock routing to specialised sub-networks, and Multi-Head Self-Attention for global cross-timestep dependencies.

2. The **Dual Hypergraph** adds a Temporal-Causal Hypergraph (TCH) — connecting stocks with consistent directional lead-lag patterns — and a Global Probabilistic Hypergraph (GPH) — grouping stocks by latent thematic membership via soft probabilistic clustering.

---

## 2. Architecture

```
Input (N × T × 4 OHLC)
        │
   Linear Embed (in_dim → D)
        │
 ┌──────┴────────────────────────────────────────────────────────────┐
 │                    MaGNet Temporal Encoder                        │
 │                                                                   │
 │   BiGRU (fwd + bwd context)                                       │
 │      │                                                            │
 │   Sparse MoE  (route each stock to 1-of-K expert sub-networks)   │
 │      │                                                            │
 │   Multi-Head Self-Attention  (cross-timestep global context)      │
 └──────┬────────────────────────────────────────────────────────────┘
        │  Z_temp  (N × T × D)
        │
 ┌──────┴──────────────────────┬──────────────────────────────────────┐
 │  Path A: TCH                │  Path B: Co-Movement                 │
 │                             │                                       │
 │  Temporal-Causal Hypergraph │  PosGAT(pos_adj)   → h_pos           │
 │  (asynchronous lead-lag)    │  NegGAT(neg_adj)   → h_neg           │
 │  → h_causal                 │  GlobalProbHG(GPH) → h_gph           │
 └──────┬──────────────────────┴──────────────────────────────────────┘
        │
 Semantic Attention Fusion  (4 streams: h_causal, h_pos, h_neg, h_gph)
        │
   PairNorm-SI  (prevents over-smoothing)
        │
  Linear(D → 1)
        │
 Predicted Return  (N × 1)
```

---

## 3. Loss Function

The training objective is a three-term composite loss optimised jointly:

```
L = w_mse  × (MSE / σ_return²)
  + w_ic   × (1 − Spearman IC)
  + w_disp × Dispersion Penalty
```

| Term | Purpose |
|---|---|
| **Normalised MSE** | Prediction accuracy; normalised to O(1) to balance with IC term |
| **1 − Spearman IC** | Directly optimises cross-sectional ranking quality (differentiable soft-rank) |
| **Dispersion Penalty** | Keeps `pred_std / target_std` within `[r_min, r_max]`; prevents collapse |

The IC weight is ramped from 0 → `--ic-weight` over `--ic-warmup-epochs` epochs to allow MSE fitting to stabilise before ranking pressure is applied.

---

## 4. Quick Start

### Install

```bash
# Requires CUDA 12.4; uses uv for reproducible installs
pip install uv
cd THGNN_MaGNet
uv sync
```

### Train

```bash
python train_hybrid.py \
  --embed-dim 64 \
  --train-start-date 2015-01-01 --train-end-date 2023-12-31 \
  --test-start-date  2024-01-01 --test-end-date  2026-02-28 \
  --epochs 80 --lr 1e-4 --patience 15 --ic-warmup-epochs 5
```

### Backtest

```bash
python backtest_hybrid.py \
  --checkpoint ../THGNN/data/model_saved/2024-12-31_hybrid_best.dat \
  --start-date 2024-01-01 --end-date 2026-02-28 --top-k 5
```

### Live Prediction Charts

```bash
python plot_live_predictions.py \
  --start-date 2026-04-01 --end-date 2026-04-10 --top-n 5
```

---

## 5. Hyperparameter Reference

### `train_hybrid.py`

| Argument | Default | Notes |
|---|---|---|
| `--embed-dim` | `32` | Model-wide embedding dimension; `64` recommended on ≥ 12 GB VRAM |
| `--num-mage-layers` | `2` | Number of MaGNet encoder blocks stacked |
| `--num-experts` | `4` | Number of MoE experts (each a small FFN) |
| `--num-heads` | `4` | Multi-head attention heads in temporal encoder |
| `--epochs` | `60` | Maximum training epochs; 80 recommended |
| `--lr` | `1e-4` | AdamW learning rate |
| `--weight-decay` | `1e-3` | AdamW L2 regularisation |
| `--patience` | `15` | Early stopping patience (epochs without val IC improvement) |
| `--ic-warmup-epochs` | `5` | Epochs over which IC weight ramps 0 → full |
| `--mse-weight` | `1.0` | Normalised MSE term weight |
| `--ic-weight` | `0.2` | Spearman IC regularisation weight |
| `--dispersion-weight` | `0.1` | Spread-ratio penalty weight |
| `--dropout` | `0.3` | Dropout rate applied throughout |

### `backtest_hybrid.py`

| Argument | Default | Notes |
|---|---|---|
| `--checkpoint` | auto-detect | Path to `.dat` checkpoint |
| `--start-date` | `2024-01-01` | Backtest start |
| `--end-date` | `2026-02-28` | Backtest end |
| `--top-k` | `5` | Number of long/short positions |
| `--transaction-cost` | `0.001` | One-way cost per trade (10 bps) |

---

## 6. Checkpoints

Saved to `../THGNN/data/model_saved/`:

| File | embed_dim | Train End | Notes |
|---|:---:|---|---|
| `2024-12-31_hybrid_best.dat` | 64 | Dec 2024 | **Best result** — use for inference |
| `2023-12-29_hybrid_best.dat` | 32 | Dec 2023 | Baseline comparison |

---

## 7. File Reference

| File | Purpose |
|---|---|
| `model/hybrid_model.py` | Complete hybrid architecture definition |
| `train_hybrid.py` | Training loop with IC-ranked loss + MaGNet warmup schedule |
| `backtest_hybrid.py` | Vectorised backtest engine with transaction costs |
| `data_loader.py` | Shared data pipeline (reads THGNN graph `.pkl` files) |
| `plot_live_predictions.py` | Inference + 4-chart visualisation for a date range |
| `ARCHITECTURE.md` | In-depth design documentation with design decisions |
| `pyproject.toml` | Dependencies (uv + CUDA 12.4 torch + torch-geometric) |
