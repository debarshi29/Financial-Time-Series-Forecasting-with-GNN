# Temporal and Heterogeneous Graph Neural Network (THGNN) for Financial Time Series Prediction

A PyTorch implementation of THGNN applied to the **Nifty 50** index. The model forecasts next-day cross-sectional stock returns by jointly encoding temporal patterns through a GRU and relational structure through heterogeneous graph attention over positive and negative correlation edges.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Model Architecture](#2-model-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Training: Criteria and Loss](#4-training-criteria-and-loss)
5. [Quick Start](#5-quick-start)
6. [Configuration Reference](#6-configuration-reference)
7. [Citing](#7-citing)

---

## 1. Project Structure

```
THGNN/
├── model/
│   └── Thgnn.py               # StockHeteGAT, GraphAttnMultiHead, GraphAttnSemIndividual, PairNorm
├── trainer/
│   └── trainer.py             # extract_data, train_epoch, eval_epoch helpers (legacy)
├── utils/
│   ├── download_market_data.py   # Step 1: download and preprocess OHLCV from Yahoo Finance
│   ├── generate_relation.py      # Step 2: build per-day pairwise correlation matrices
│   └── generate_data.py          # Step 3: assemble graph samples (features + adjacency + labels)
├── data_loader.py             # AllGraphDataSampler (PyTorch Dataset)
├── train_ic_ranked.py         # PRIMARY training script (MSE + IC + dispersion loss)
├── rebuild_graph_data.py      # Utility: rebuild data_train_predict/ with a custom adj threshold
├── data/
│   ├── nifty50.pkl            # Processed market data (output of download_market_data.py)
│   ├── relation/              # Per-day correlation CSVs (output of generate_relation.py)
│   ├── data_train_predict/    # Daily graph samples as .pkl files (output of generate_data.py)
│   ├── daily_stock/           # Per-day stock metadata CSVs
│   ├── model_saved/           # Checkpoints saved during training
│   └── plots/                 # Loss curve PNGs
└── requirements.txt
```

---

## 2. Model Architecture

The model is defined in [model/Thgnn.py](model/Thgnn.py) as `StockHeteGAT`.

### 2.1 Overview

For each trading day, the input is a graph of **N stocks** (N = 50 for Nifty 50). Each stock node carries a **20-day window of 6 features** `[open, high, low, close, to, vol]` (all as percentage-change returns). The model outputs a scalar predicted return for each stock.

The forward pass has four stages:

```
Input (N, 20, 6)
    |
    v
[1] Feature Normalization
    LayerNorm(6) per timestep  +  cross-sectional demeaning
    |
    v
[2] Temporal Encoding  --  GRU
    (N, 20, 6) --> hidden state (N, hidden_dim)
    |
    v
[3] Heterogeneous Graph Attention  --  pos-GAT and neg-GAT in parallel
    + self-projection MLP
    |
    v
[4] Semantic Attention  --  GraphAttnSemIndividual
    Soft-weights the three embeddings (self / pos-neighbor / neg-neighbor)
    |
    v
[5] PairNorm (PN-SI)
    |
    v
[6] Linear predictor
    (N, hidden_dim) --> (N, 1)  predicted next-day return
```

---

### 2.2 Input Normalization

**LayerNorm** (`nn.LayerNorm(in_features=6)`) is applied per timestep before the GRU. This prevents the volume and turnover channels (`to`, `vol`) — which can spike up to 100× the daily average — from dominating the GRU gate activations.

**Cross-sectional demeaning** then subtracts the market-wide average across all N stocks at each (timestep, channel):

```python
inputs = self.input_norm(inputs)          # (N, 20, 6)
inputs = inputs - inputs.mean(dim=0, keepdim=True)  # remove market beta
```

This aligns the representation with the cross-sectional IC objective: the model focuses on *relative* stock behavior rather than absolute market-wide moves.

---

### 2.3 Temporal Encoding — GRU

```python
self.encoding = nn.GRU(
    input_size=in_features,   # 6
    hidden_size=hidden_dim,   # default 128
    num_layers=num_layers,    # default 1
    batch_first=True,
)
```

Each stock's 20-day feature sequence is encoded independently. Only the **final hidden state** `h_T` is used, giving a per-stock embedding of shape `(N, hidden_dim)`. Dropout is applied after the GRU output.

---

### 2.4 Heterogeneous Graph Attention — `GraphAttnMultiHead`

Two separate multi-head GAT layers operate on the same node embeddings but over different adjacency matrices:

| Layer | Adjacency | Semantics |
|-------|-----------|-----------|
| `pos_gat` | `pos_adj` (corr > threshold) | Stocks that move together |
| `neg_gat` | `neg_adj` (corr < -threshold) | Stocks that move inversely |

**Attention mechanism** (per head `k`):

```
support_k = W · h            # (N, out_features) linear projection
f_1 = support_k · w_u       # (1, N) source scores
f_2 = support_k · w_v       # (N, 1) target scores
logits = f_1 + f_2           # (N, N) attention logits (additive)
attn = softmax(LeakyReLU(logits) * adj_mask)  # sparse masked softmax
output_k = attn · support_k  # attended neighbor aggregation
```

Heads are concatenated: output shape is `(N, num_heads * out_features)`. A residual skip-connection (`self.project`) adds the original embedding back.

Default: `num_heads=8`, `out_features=32` → `256`-dim output per GAT.

---

### 2.5 Self + Neighbor Projection MLPs

After graph attention, three parallel linear projections (each followed by ReLU and dropout) map all embeddings into the same `hidden_dim`-dimensional space:

```python
support     = drop(relu(mlp_self(h_T)))         # self  (N, hidden_dim)
pos_support = drop(relu(mlp_pos(pos_out)))       # positive neighbors
neg_support = drop(relu(mlp_neg(neg_out)))       # negative neighbors
```

The three are stacked into a tensor of shape `(N, 3, hidden_dim)`.

---

### 2.6 Semantic Attention — `GraphAttnSemIndividual`

A lightweight two-layer MLP with Tanh activation computes a **per-node, per-semantic** soft weight:

```python
w = Linear(hidden_dim, hidden_size) -> Tanh -> Linear(hidden_size, 1)
beta = softmax(w, dim=1)          # (N, 3, 1) weights over 3 semantics
embedding = (beta * stack).sum(1) # (N, hidden_dim) weighted combination
```

Each stock independently learns which information source (self / positive peers / negative peers) to attend to.

---

### 2.7 PairNorm

`PairNorm(mode='PN-SI')` prevents over-smoothing from graph aggregation:

```
x = x / ||x||_row  * scale  -  col_mean
```

Individual row-normalization followed by column-mean subtraction keeps embeddings separated in representation space regardless of graph density.

---

### 2.8 Predictor

A single linear layer maps each stock's `hidden_dim`-dimensional embedding to a scalar:

```python
self.predictor = nn.Linear(hidden_dim, predictor_out_dim)  # (N, 1)
```

No activation is used for regression (return prediction). All linear layers are initialized with Xavier uniform at `gain=0.02` to keep initial predictions in a small range consistent with daily return magnitudes (~±2%).

---

### 2.9 Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in_features` | 6 | Number of input channels per timestep |
| `hidden_dim` | 128 | GRU hidden size and MLP output dimension |
| `num_layers` | 1 | Number of GRU layers |
| `num_heads` | 8 | Number of attention heads in each GAT |
| `out_features` | 32 | Per-head output dimension in each GAT |
| `predictor_out_dim` | 1 | Output dimension (1 = single-horizon return) |
| `predictor_activation` | None | Optional `'sigmoid'` for classification tasks |
| `dropout` | 0.3 | Dropout rate applied after GRU and each MLP |

---

## 3. Data Pipeline

All scripts are in [utils/](utils/). Run them in order.

### Step 1 — Download Market Data

**Script:** [utils/download_market_data.py](utils/download_market_data.py)
**Output:** `data/nifty50.pkl`

Downloads adjusted OHLCV data for all 50 Nifty constituents from Yahoo Finance and engineers percentage-change features:

| Column | Description |
|--------|-------------|
| `dt` | Trading date (YYYY-MM-DD) |
| `code` | Ticker symbol (e.g. `RELIANCE.NS`) |
| `open` | Daily open % change |
| `high` | Daily high % change |
| `low` | Daily low % change |
| `close` | Daily close % change |
| `to` | Turnover (price × volume) % change |
| `vol` | Volume % change |
| `label` | Next-day close % change (forward return, used as training target) |

Adjustment logic: raw prices are multiplied by the `Adj Close / Close` ratio to correct for splits and dividends before computing returns.

```bash
# Standard run — all 50 Nifty tickers, 2015 to present
python utils/download_market_data.py --start 2015-01-01

# Specify an explicit end date
python utils/download_market_data.py --start 2015-01-01 --end 2024-01-01

# Also export per-ticker CSV files with RSI and MACD indicators
python utils/download_market_data.py --start 2015-01-01 --csv-dir data/nifty50_csv

# Custom ticker list
python utils/download_market_data.py --tickers "RELIANCE.NS,TCS.NS,INFY.NS" --start 2020-01-01

# Back off if Yahoo Finance rate-limits
python utils/download_market_data.py --start 2015-01-01 --pause 3 --max-retries 5
```

---

### Step 2 — Generate Relation Matrices

**Script:** [utils/generate_relation.py](utils/generate_relation.py)
**Output:** `data/relation/YYYY-MM-DD.csv` (one file per trading day)

Computes a pairwise Pearson correlation matrix over the previous `--window` trading days for all stocks present on that date. The correlation is computed across all six feature channels (high, low, close, open, to, vol) and averaged.

Each output CSV is an `N × N` matrix with stock tickers as row/column indices and correlation values in `[-1, 1]`.

```bash
# Generate a relation matrix for every trading day in the dataset
python utils/generate_relation.py --data-path data/nifty50.pkl --relation-dir data/relation --window 20

# Restrict to a date range
python utils/generate_relation.py --data-path data/nifty50.pkl --relation-dir data/relation \
    --start-date 2020-01-01 --end-date 2024-01-01

# Use multiprocessing for faster computation
python utils/generate_relation.py --data-path data/nifty50.pkl --relation-dir data/relation \
    --processes 4
```

---

### Step 3 — Build Graph Samples

**Script:** [utils/generate_data.py](utils/generate_data.py)
**Output:** `data/data_train_predict/YYYY-MM-DD.pkl` and `data/daily_stock/YYYY-MM-DD.csv`

For each trading day that has a corresponding relation CSV, assembles a complete graph sample:

- **features** `(N, 20, 6)` — 20-day window of OHLCV returns for each stock
- **labels** `(N, 3)` — forward returns for t+1, t+2, t+3 days
- **pos_adj** `(N, N)` — binary adjacency: 1 if correlation > `pos_threshold`
- **neg_adj** `(N, N)` — binary adjacency: 1 if correlation < `-neg_threshold`
- **mask** — list of `True` values (all stocks present)

```bash
python utils/generate_data.py \
    --data-path data/nifty50.pkl \
    --relation-dir data/relation \
    --output-dir data/data_train_predict \
    --daily-stock-dir data/daily_stock

# Sparser graph: only connect stocks with correlation > 0.3
python utils/generate_data.py \
    --data-path data/nifty50.pkl \
    --relation-dir data/relation \
    --output-dir data/data_train_predict \
    --pos-threshold 0.3 \
    --neg-threshold 0.3
```

> **Adjacency threshold note:** The default threshold of 0.1 produces ~49% edge density (near fully-connected). A threshold of 0.3 reduces this to ~15%, giving the GAT meaningful structure to exploit. Use `rebuild_graph_data.py` to rebuild existing samples with a different threshold without re-running the full pipeline.

```bash
python rebuild_graph_data.py --threshold 0.3   # recommended
python rebuild_graph_data.py --threshold 0.1   # restore original
```

---

## 4. Training: Criteria and Loss

**Use [train_ic_ranked.py](train_ic_ranked.py) for all training.**

This is the primary training script. It implements a composite loss that directly optimizes for cross-sectional ranking quality (Information Coefficient) alongside reconstruction accuracy.

### 4.1 Composite Loss

The total loss per training step is:

```
L = w_mse * MSE  +  w_ic * (1 - IC)  +  w_disp * max(0, sigma_min - sigma_pred)
```

| Term | Weight arg | Default | Purpose |
|------|-----------|---------|---------|
| `MSE` | `--mse-weight` | `1.0` | Penalizes absolute prediction error |
| `1 - IC` | `--ic-weight` | `0.35` | Maximizes cross-sectional Spearman rank correlation |
| Dispersion penalty | `--dispersion-weight` | `0.2` | Prevents prediction collapse to near-zero variance |

**IC (Information Coefficient)** is the cross-sectional **Spearman rank correlation** between predicted and actual returns across all N stocks on a single day. The training objective uses a **differentiable soft-rank approximation**: each stock's soft rank is computed via pairwise sigmoid comparisons (`rank_i ≈ Σ_j sigmoid((pred_i − pred_j) / τ)`, τ = 0.01), making the Spearman IC fully differentiable. This is more robust than raw Pearson IC, which can be dominated by a single outlier stock with an extreme return on a given day. An IC of 0.05 is considered practically useful; IC > 0.10 is strong.

**Dispersion penalty** fires when the standard deviation of predictions falls below `--min-dispersion-ratio` (default 0.2) times the target standard deviation. This prevents the degenerate solution where the model predicts the same value for all stocks.

**Directional accuracy** is tracked as a monitoring metric only — it is not part of the loss. It is computed inside `torch.no_grad()` and reported in the progress bar.

### 4.2 Single-Horizon Training

The label tensor has 3 horizons `(N, 3)` but **only one horizon is used for training**, controlled by `--target-horizon`:

| Value | Target |
|-------|--------|
| `0` (default) | t+1 next-day return |
| `1` | t+2 return |
| `2` | t+3 return |

Training on multiple horizons simultaneously was found to hurt IC because h0 and h2 are negatively correlated (corr ≈ -0.24), creating conflicting gradients.

### 4.3 Optimizer and Regularization

| Component | Setting |
|-----------|---------|
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-3` |
| LR schedule | CosineAnnealingLR over total epochs |
| Gradient clipping | `max_norm=1.0` |
| Dropout | `0.3` (applied after GRU and each MLP) |

### 4.4 Model Selection

The best checkpoint is saved to `data/model_saved/<pre_data>_icrank_best.dat` and updated whenever **validation IC improves** (with MSE as a tiebreaker). Early stopping is triggered after `--patience` epochs without improvement (default 15).

The checkpoint stores:

```python
{
    "model":   model.state_dict(),
    "epoch":   int,
    "val_ic":  float,
    "val_mse": float,
    "config":  vars(args),     # full argument namespace
    "split":   split.__dict__, # train/val/test index ranges
}
```

### 4.5 Data Splits

Splits are defined by date strings and resolved to file indices at runtime:

| Argument | Default | Role |
|----------|---------|------|
| `--train-start-date` | `2015-01-01` | First training sample |
| `--train-end-date` | `2023-12-29` | Last training sample (no hard ceiling — any date with available data is valid) |
| `--val-start-date` | `2024-01-01` | First validation sample |
| `--val-end-date` | `2024-12-31` | Last validation sample |
| `--test-start-date` | `2025-01-01` | First test sample |
| `--test-end-date` | `2026-02-28` | Last test sample |

A **purge gap** is automatically applied between training and validation: the last `horizon - 1` training days are dropped to prevent label leakage (labels extend `horizon` days into the future).

### 4.6 Output

After training, the script prints final metrics from the best checkpoint across all three splits and saves a loss curve plot to `data/plots/<pre_data>_icrank_loss_curve.png`.

```
Training complete.
Best epoch: 18
Best val IC: 0.0293
Best val MSE: 0.000412
Directional accuracy | train=0.5321, val=0.5082, test=0.5001
Saved loss plot: data/plots/2023-12-29_icrank_loss_curve.png
```

---

## 5. Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Full pipeline from scratch

```bash
# 1. Download data
python utils/download_market_data.py --start 2015-01-01

# 2. Build correlation graphs
python utils/generate_relation.py --data-path data/nifty50.pkl --relation-dir data/relation --window 20

# 3. Build graph samples (recommended threshold=0.3 for meaningful graph structure)
python utils/generate_data.py \
    --data-path data/nifty50.pkl \
    --relation-dir data/relation \
    --output-dir data/data_train_predict \
    --daily-stock-dir data/daily_stock \
    --pos-threshold 0.3 --neg-threshold 0.3

# 4. Train
python train_ic_ranked.py
```

### Train only (data already prepared)

```bash
python train_ic_ranked.py \
    --train-start-date 2015-01-01 \
    --train-end-date 2023-12-29 \
    --val-start-date 2024-01-01 \
    --val-end-date 2024-12-31 \
    --test-start-date 2025-01-01 \
    --test-end-date 2026-02-28 \
    --epochs 60 \
    --lr 1e-4 \
    --hidden-dim 128 \
    --num-heads 8 \
    --dropout 0.3 \
    --ic-weight 0.35 \
    --mse-weight 1.0 \
    --dispersion-weight 0.2
```

### Rebuild adjacency with a different threshold (no re-download needed)

```bash
python rebuild_graph_data.py --threshold 0.3
```

---

## 6. Configuration Reference

### `train_ic_ranked.py` arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data/data_train_predict` | Directory containing graph sample `.pkl` files |
| `--model-dir` | `data/model_saved` | Directory to save checkpoints |
| `--train-start-date` | `2015-01-01` | Start of training split |
| `--train-end-date` | `2023-12-29` | End of training split |
| `--val-start-date` | `2024-01-01` | Start of validation split |
| `--val-end-date` | `2024-12-31` | End of validation split |
| `--test-start-date` | `2025-01-01` | Start of test split |
| `--test-end-date` | `2026-02-28` | End of test split |
| `--epochs` | `60` | Maximum number of training epochs |
| `--lr` | `1e-4` | AdamW learning rate |
| `--weight-decay` | `1e-3` | AdamW L2 regularization |
| `--dropout` | `0.3` | Dropout rate |
| `--hidden-dim` | `128` | GRU hidden size and MLP width |
| `--num-heads` | `8` | Number of attention heads per GAT |
| `--num-layers` | `1` | Number of GRU layers |
| `--out-features` | `32` | Per-head output features in GAT |
| `--mse-weight` | `1.0` | Weight on MSE term in composite loss |
| `--ic-weight` | `0.35` | Weight on IC loss term (1 − soft Spearman rank corr) |
| `--dispersion-weight` | `0.2` | Weight on dispersion penalty |
| `--min-dispersion-ratio` | `0.2` | Minimum pred_std / target_std before penalty fires |
| `--target-horizon` | `0` | Label horizon index to train on (0=next-day) |
| `--patience` | `15` | Early stopping patience (epochs without val IC improvement) |
| `--seed` | `42` | Random seed |

---

## 7. Citing

If you find THGNN useful for your research, please consider citing the original paper:

```bibtex
@inproceedings{Xiang2022Temporal,
  author    = {Xiang, Sheng and Cheng, Dawei and Shang, Chencheng and Zhang, Ying and Liang, Yuqi},
  title     = {Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction},
  year      = {2022},
  isbn      = {9781450392365},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3511808.3557089},
  booktitle = {Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
  pages     = {3584--3593},
  series    = {CIKM '22}
}
```
