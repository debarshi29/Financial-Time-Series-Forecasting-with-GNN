# Hybrid THGNN × MaGNet — Cascaded-Parallel Architecture

## Overview

The model fuses two GNN approaches for cross-sectional stock return prediction.
It takes `(N, T, F)` as input — N stocks, T timesteps, F features per timestep —
and produces `(N, 1)` return predictions used for portfolio ranking.

```
Input (N, T, F)
  → LayerNorm + cross-sectional demean
  → Linear(F → D)                              [embed_dim=64 default]
  → MAGEBlock × num_mage_layers                [temporal encoder]
      BiGRU(fwd+bwd) → Gating → SparseMoE → MHA
                                               → Z_temp (N, T, D)
  │
  ├─── Path A: TCH(Z_temp)         → h_causal  (N, D)   [lead-lag causality]
  │    Flatten (T·N, D) → Causal MHA (block mask)
  │    → ReTanh FFN → H_TCH (T·N, M1) → Hypergraph conv
  │    → reshape (N, T, D) → slice last timestep
  │
  └─── Path B: h_temp = Z_temp[:, -1, :]       (N, D)
          ├─── PosGAT(h_temp, pos_adj)  → h_pos (N, D)  [explicit co-movement]
          ├─── NegGAT(h_temp, neg_adj)  → h_neg (N, D)  [explicit inverse-movement]
          └─── GPHypergraph(h_temp)     → h_gph (N, D)  [latent macro themes]

  4-stream Semantic Attention Fusion(h_causal, h_pos, h_neg, h_gph)
  → PairNorm-SI
  → Linear(D → 1)                              (N, 1) predictions
```

---

## Key Components

### 1. MAGEBlock (from MaGNet) — Temporal Encoder

**File:** `model/hybrid_model.py`, class `MAGEBlock`

Each block processes the `(N, T, D)` sequence through three sequential sub-layers:

**a) Bidirectional GRU + Learnable Gating**
- BiGRU processes all T timesteps bidirectionally, producing `z_fwd` and `z_bwd`
- Soft gate blends directions: `gate = σ(W_f·z_fwd + W_b·z_bwd)`
- Fused state: `z_G = gate * z_fwd + (1 - gate) * z_bwd`
- Residual + LayerNorm applied after

**b) Sparse Mixture-of-Experts (SparseMoE)**
- Top-1 routing: each token is assigned to one of `num_experts` 2-layer FFNs via argmax of gating softmax
- All experts are computed to ensure full gradient flow; output is weighted by routing probability
- Captures market-regime specialisation (different experts for different volatility regimes)
- Residual + LayerNorm applied after

**c) Multi-Head Self-Attention (MHA)**
- Temporal self-attention per stock over T timesteps
- Captures long-range temporal dependencies within each stock's own history
- Residual + LayerNorm applied after

> **Note:** This is a **BiGRU-lite** variant of MaGNet's original Mamba SSM design,
> dropping the `mamba-ssm` C++ dependency while preserving the same information flow.

---

### 2. TemporalCausalHypergraph / TCH (from MaGNet) — Lead-Lag Causal Discovery

**File:** `model/hybrid_model.py`, class `TemporalCausalHypergraph`

Captures **asynchronous, cross-stock causal relationships** — the "ripple effects" where
a shock in stock A at time t propagates to stock B at a later time t+k. This is the market
force that MAGE's temporal MHA misses: MHA fuses timesteps *per stock*, while TCH
builds causal paths *across stocks and time* simultaneously.

**Steps:**
1. **Flatten to (T·N, D):** every (time, stock) pair becomes an independent node, ordered
   `(t₀·S₀, …, t₀·Sₙ₋₁, t₁·S₀, …, tₜ₋₁·Sₙ₋₁)`
2. **Causal MHA with upper-triangular block mask:** node at time t can only attend to
   nodes at t' ≤ t — strictly prevents future-data leakage
3. **Two-layer ReTanh FFN → H_TCH ∈ (T·N, M1):** builds the sparse incidence matrix;
   each column is one causal hyperedge connecting multiple (time, stock) nodes
4. **Efficient hypergraph convolution:**
   `Z' = ELU(H · (H^T · Proj(Z))) + Z` — computed in O(T·N · M1 · D) to avoid
   materialising the full (T·N)² matrix
5. **Reshape + slice:** `(T·N, D) → (N, T, D) → h_causal = output[:, -1, :]`

**Why TCH and MAGE are complementary, not redundant:**
- MAGE-MHA: attends over T timesteps *for each stock independently* — captures intra-stock long-range patterns
- TCH causal MHA: attends over T×N time-stock pairs — captures inter-stock causal chains (e.g., Reliance at t₋₂ → ONGC at t₀)

---

### 3. Pos/Neg GAT (from THGNN) — Explicit Correlation Streams

**File:** `model/hybrid_model.py`, class `GraphAttnMultiHead`

Two separate multi-head additive GATs run on pre-computed adjacency matrices:
- **pos_adj**: positive-correlation edges (stocks that historically move together)
- **neg_adj**: negative-correlation edges (stocks that historically move inversely)

Both operate on `h_temp = Z_temp[:, -1, :]` — the pristine, TCH-independent current state —
so the explicit statistical correlations are evaluated on un-mixed node representations.

**Attention mechanism:**
```
f_1 = W_u · (W · h)     f_2 = W_v · (W · h)
attention = softmax(LeakyReLU(f_1 + f_2) ⊙ adj)
output = attention · (W · h)
```

Output dimension: `gat_heads × gat_out_features`, projected back to D via MLP.

---

### 4. Global Probabilistic Hypergraph / GPH (from MaGNet)

**File:** `model/hybrid_model.py`, class `GPHypergraph`

Discovers **instantaneous latent market themes** (sector rallies, risk-off events, factor
rotations) that the explicit pairwise GATs cannot capture because they are constrained to
pre-computed Pearson correlation pairs.

**Steps:**
1. **Soft incidence matrix:** `H = softmax(ReTanh(Linear(h_temp)))` of shape `(N, M)`
   - Column-softmax: each hyperedge is a probability distribution over N stocks
2. **JSD-based importance weights:** hyperedges more distinct from others receive higher
   weight `w` (suppresses redundant hyperedges capturing the same theme)
3. **Efficient hypergraph convolution:** `z = ELU(H · diag(w) · (H^T · Proj(h))) + h`

Also operates on `h_temp` (Path B), maintaining orthogonality with TCH (Path A).

---

### 5. Semantic Attention Fusion (from THGNN)

**File:** `model/hybrid_model.py`, class `GraphAttnSemIndividual`

Stacks all four streams as `(N, 4, D)` and learns per-stream soft weights:
```
β = softmax(Linear(tanh(Linear([h_causal, h_pos, h_neg, h_gph]))))
fused = Σ β_i · h_i
```

The model dynamically decides each trading day which view is most predictive:
- `h_causal`: "Which stocks sent causal shocks to me in recent days?"
- `h_pos`:    "What are my historically correlated peers doing?"
- `h_neg`:    "What are my historically inverse peers doing?"
- `h_gph`:    "What macro theme am I currently grouped into?"

Followed by **PairNorm-SI** to prevent over-smoothing across the N-stock graph.

---

### 6. Design Rationale — What Was Kept, Dropped, and Reinstated

| Component | Decision | Reason |
|---|---|---|
| TCH (Temporal-Causal Hypergraph) | **Reinstated** | Captures asynchronous lead-lag causal ripples across stocks — orthogonal to MAGE's per-stock temporal MHA |
| MAGE (BiGRU-lite) | **Kept** | Per-stock temporal encoding with market-regime MoE specialisation |
| Pos/Neg GAT | **Kept** | Explicit, pre-computed correlation inductive bias complements learned hyperedges |
| GPH | **Kept** | Captures latent thematic groupings beyond pairwise correlations |
| 2D Feature Attention | **Dropped** | MAGE's per-stock MHA already captures cross-timestep feature dependencies |
| IC-ranked composite loss | **Kept** | Better for cross-sectional portfolio ranking than MaGNet's original BCE |
| Self-MLP stream | **Replaced by TCH** | Plain MLP projection provides no relational information; TCH adds meaningful causal structure |

**Why TCH and GPH are kept strictly parallel (not cascaded):**
Feeding TCH output into GPH would cause over-smoothing — GPH would cluster already-mixed
causal representations, blurring the distinct stock profiles needed for sharp hyperedge
discovery. Keeping them as independent parallel paths ensures each discovers its own
orthogonal market force on pristine node representations.

---

## Loss Function

**File:** `train_hybrid.py`, function `composite_loss`

Three weighted terms optimised jointly:

| Term | Default Weight | Formula |
|---|---|---|
| **MSE** | `mse_weight = 0.7` | `MSE(pred_cs, target_cs) / σ²_return` — cross-sectionally demeaned, normalised by typical return scale |
| **IC Loss** | `ic_weight = 0.5` | `1 - soft_Spearman_IC` — differentiable soft-ranks via sigmoid approximation |
| **Dispersion Penalty** | `dispersion_weight = 0.3` | `ReLU(min_ratio - σ_pred/σ_target) + ReLU(σ_pred/σ_target - max_ratio)` — keeps prediction spread in `[0.5×, 2.0×]` of target spread |

**Training refinements:**
- **IC warmup:** IC weight ramps from 0 → full over `ic_warmup_epochs=3`, letting MSE stabilise first
- **Temperature annealing:** soft-rank temperature decays as `max(0.02, 0.2 × 0.95^epoch)`, sharpening toward exact Spearman ranks over training

**Logged metrics (no-gradient):**
- `rank_ic`: exact Spearman IC on predictions vs. targets
- `dir_acc`: directional accuracy (fraction of correct up/down calls)
- `pred_std` / `target_std`: spread ratio monitoring

---

## Training Setup

**File:** `train_hybrid.py`, function `main`

| Setting | Default | Notes |
|---|---|---|
| Optimizer | AdamW | `lr=5e-4`, `weight_decay=5e-5` |
| LR Schedule | Cosine with warmup | 2-epoch linear warmup then cosine decay |
| Gradient clipping | `max_norm=1.0` | Applied every step |
| Early stopping | `patience=10` | Monitors test Spearman IC; tie-broken by MSE |
| Checkpoint | Best by test rank-IC | Reloaded for final evaluation |
| Data format | Daily `.pkl` snapshots | One graph per trading day from THGNN sibling directory |
| Batch size | 1 | Each batch is one trading day's full stock graph |

### Date-based Train/Test Split

Files are sorted by date; `compute_split_indices` maps date strings to file indices.
The split enforces strict ordering: `train_start < train_end ≤ test_start < test_end`.

Default split:
- **Train:** 2015-01-01 → 2024-12-31
- **Test:** 2025-01-01 → 2026-12-31

---

## Default Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `in_features` | 12 | Input feature dimension per stock per timestep |
| `embed_dim` | 64 | Core embedding dimension D used throughout |
| `num_mage_layers` | 1 | Number of stacked MAGE blocks |
| `num_moe_experts` | 4 | Experts per SparseMoE layer |
| `num_mha_heads` | 2 | Heads in MAGE's temporal self-attention |
| `gat_heads` | 8 | Heads in Pos/Neg GAT |
| `gat_out_features` | 8 | Per-head output dim in GAT (total GAT dim = 64) |
| `num_hyper_edges` | 32 | Hyperedges M in GPH (increase for larger universes) |
| `num_tch_hyper_edges` | 32 | Causal hyperedges M1 in TCH |
| `num_tch_heads` | 4 | Heads in TCH's causal MHA (must divide `embed_dim`) |
| `dropout` | 0.1 | Applied throughout |
| `epochs` | 80 | Maximum training epochs |

---

## Computational Notes

**TCH memory and compute:**
- The causal mask is `(T·N, T·N)` — for T=20, N=50 (Nifty50), this is `1000×1000 = 4 MB`
- The causal MHA computes `O((T·N)² · D)` attention — for TN=1000, D=64: ~64M FLOPs per sample
- Hypergraph conv uses the factored form `H · (H^T · Z)` — `O(T·N · M1 · D)` instead of `O((T·N)² · D)`
- For larger universes (N=300, T=20 → TN=6000), consider reducing `num_tch_heads` or increasing `embed_dim` divisibility

---

## Usage

```bash
# Train on existing THGNN Nifty50 data (default paths):
python train_hybrid.py

# Custom date range:
python train_hybrid.py --train-start-date 2018-01-01 --train-end-date 2023-12-31 \
    --test-start-date 2024-01-01 --test-end-date 2024-12-31

# Larger embed dim for N=300 universe:
python train_hybrid.py --embed-dim 128 --num-hyper-edges 64 --num-tch-hyper-edges 64

# Custom TCH settings:
python train_hybrid.py --num-tch-hyper-edges 48 --num-tch-heads 4

# Custom data directory:
python train_hybrid.py --data-dir /path/to/data_train_predict
```
