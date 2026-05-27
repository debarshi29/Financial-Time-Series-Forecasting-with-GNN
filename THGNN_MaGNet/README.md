# THGNN × MaGNet — Hybrid Temporal-Hypergraph Stock Forecasting Model

This module implements the **main thesis contribution**: a hybrid architecture that extends the baseline THGNN with a MaGNet-style temporal encoder, a Temporal Causal Hypergraph (TCH) path, and a latent Graph-based Hypergraph (GPH) path, all fused via semantic attention.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a full design explanation.

---

## Architecture Summary

```
Input (N × T × 4 OHLC)
        │
   Linear Embed (→ D)
        │
 ┌──────┴──────────────────────────────────────────────────┐
 │              MaGNet Temporal Encoder                     │
 │  BiGRU  ──▶  Sparse MoE  ──▶  Multi-Head Attention      │
 │  (fwd+bwd context)  (expert routing)  (cross-timestep)  │
 └──────┬──────────────────────────────────────────────────┘
        │  Z_temp (N × T × D)
        │
 ┌──────┴──────────────┬──────────────────────────────────┐
 │  Path A: TCH        │  Path B: Co-Movement             │
 │                     │                                   │
 │  Temporal Causal    │  PosGAT(pos_adj)  → h_pos        │
 │  Hypergraph         │  NegGAT(neg_adj)  → h_neg        │
 │  → h_causal         │  GPHypergraph()   → h_gph        │
 └──────┬──────────────┴──────────────────────────────────┘
        │
 Semantic Attention Fusion (4 streams)
        │
   PairNorm-SI
        │
  Linear(D → 1)
        │
 Predicted Return (N × 1)
```

**Loss:** `MSE + λ_ic × (1 − Spearman IC) + λ_disp × dispersion_penalty`

---

## Quick Start

```bash
# Install (requires CUDA 12.4 for full performance)
pip install uv
uv sync

# Train hybrid model
python train_hybrid.py \
  --embed-dim 64 \
  --train-start-date 2015-01-01 --train-end-date 2023-12-31 \
  --test-start-date  2024-01-01 --test-end-date  2026-02-28 \
  --epochs 80 --lr 1e-4 --patience 15 --ic-warmup-epochs 5

# Backtest with saved checkpoint
python backtest_hybrid.py \
  --checkpoint ../THGNN/data/model_saved/2024-12-31_hybrid_best.dat \
  --start-date 2024-01-01 --end-date 2026-02-28 --top-k 5

# Generate live prediction charts
python plot_live_predictions.py \
  --start-date 2026-04-01 --end-date 2026-04-10 --top-n 5
```

---

## Key Hyperparameters

| Argument | Default | Notes |
|---|---|---|
| `--embed-dim` | `32` | `64` recommended on ≥ 12 GB VRAM |
| `--num-mage-layers` | `2` | Number of MaGNet encoder blocks |
| `--num-experts` | `4` | MoE expert count |
| `--epochs` | `60` | 80 recommended |
| `--patience` | `15` | Early stopping on validation IC |
| `--ic-warmup-epochs` | `5` | IC loss ramps 0 → full over N epochs |
| `--mse-weight` | `1.0` | MSE term weight |
| `--ic-weight` | `0.2` | Spearman IC regularisation weight |
| `--dispersion-weight` | `0.1` | Spread-ratio penalty weight |

---

## Checkpoints

Saved to `../THGNN/data/model_saved/`:

| File | Notes |
|---|---|
| `2024-12-31_hybrid_best.dat` | **Best result** — embed_dim=64, trained Dec 2024 |
| `2023-12-29_hybrid_best.dat` | Earlier run for comparison |

---

## Files

| File | Purpose |
|---|---|
| `model/hybrid_model.py` | Full hybrid architecture definition |
| `train_hybrid.py` | Training with IC-ranked loss + MaGNet warmup |
| `backtest_hybrid.py` | Backtest evaluation with transaction costs |
| `data_loader.py` | Shared data pipeline (reads THGNN graph .pkl files) |
| `plot_live_predictions.py` | Inference visualisation for a date range |
| `ARCHITECTURE.md` | Detailed design documentation |
| `pyproject.toml` | Dependencies (uv + CUDA 12.4 torch) |
