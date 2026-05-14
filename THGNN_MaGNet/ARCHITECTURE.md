# Hybrid THGNN × MaGNet — Architecture Reference

## Overview

The `HybridStockModel` is a **cascaded-parallel** architecture that combines temporal
sequence modelling (MaGNet lineage) with explicit relational graph reasoning (THGNN lineage)
for cross-sectional stock return ranking.

```
Input (N, T, F)
  → LayerNorm + cross-sectional demean
  → Linear(F → D)                              [feature embedding]
  → MAGEBlock × L                              [temporal encoder]
  │
  ├─ Path A: TemporalCausalHypergraph(Z_temp)  → h_causal  (N, D)
  │
  └─ Path B (on h_temp = Z_temp[:, -1, :])
       ├─ PosGAT(h_temp, pos_adj)              → h_pos     (N, D)
       ├─ NegGAT(h_temp, neg_adj)              → h_neg     (N, D)
       └─ GPHypergraph(h_temp)                 → h_gph     (N, D)

4-stream SemanticFusion([h_causal, h_pos, h_neg, h_gph])
  → PairNorm-SI
  → Linear(D → 1)                              predictions (N, 1)
```

**Design principle**: Path A (TCH) and Path B (GAT/GPH) operate on independent
representations of the same encoder output. TCH sees the full temporal cube
`(N, T, D)` to find asynchronous lead-lag causal ripples; the GAT/GPH streams
see only the final-timestep snapshot `(N, D)` for synchronous relational signals.
This strict separation prevents the two pathways from interfering before fusion.

---

## Tensor Shapes

| Symbol | Meaning | Default |
|--------|---------|---------|
| N | Number of stocks in universe | ~350 (Nifty 500) |
| T | Lookback timesteps per sample | dataset-dependent |
| F | Input features per stock per timestep | 12 |
| D | Core embedding dimension (`embed_dim`) | 64 |
| E | MoE experts (`num_moe_experts`) | 4 |
| H_mha | Temporal MHA heads (`num_mha_heads`) | 2 |
| H_gat | GAT heads (`gat_heads`) | 8 |
| d_gat | GAT per-head output dim (`gat_out_features`) | 8 → concat D=64 |
| M | GPH hyperedges (`num_hyper_edges`) | 32 |
| M1 | TCH causal hyperedges (`num_tch_hyper_edges`) | 32 |
| H_tch | TCH causal MHA heads (`num_tch_heads`) | 4 |
| L | MAGE stack depth (`num_mage_layers`) | 1 |

---

## Input Preprocessing

```python
x = LayerNorm(F)(inputs)          # (N, T, F)  per-feature normalisation
x = x - x.mean(dim=0)            # (N, T, F)  cross-sectional demean (zero-mean at each timestep)
x = Linear(F, D)(x)              # (N, T, D)  project to embedding space
```

Cross-sectional demeaning removes the common market beta so the model
predicts **relative** return rank rather than absolute level.

---

## MAGEBlock — Temporal Encoder

Processes each stock's time series independently: `(N, T, D) → (N, T, D)`.
Stacked `L` times.

### Sub-components

#### 1. Bidirectional GRU (BiGRU)

Replaces the Mamba SSM from the original MaGNet paper. Captures long-range
temporal dependencies in both forward and backward directions.

```
gru_out = BiGRU(x)      → (N, T, 2D)
z_fwd = gru_out[:, :, :D]
z_bwd = gru_out[:, :, D:]
```

#### 2. Gating

Fuses forward and backward states with a learned gate at each timestep:

```
gate = σ(W_fwd · z_fwd + W_bwd · z_bwd)     (N, T, D)
z_G  = gate ⊙ z_fwd + (1 − gate) ⊙ z_bwd
z_G  = LayerNorm(Dropout(z_G) + x)           [residual]
```

#### 3. Sparse Mixture-of-Experts (SparseMoE)

Top-1 expert routing: each token is assigned to a single expert (argmax of
gating softmax), but all experts are evaluated to maintain gradient flow.

```
probs      = softmax(Linear(D, E)(x_flat))   [gating distribution over E experts]
assignment = argmax(probs)                    [hard routing per token]
out        = probs[assignment] · Expert_{assignment}(x_flat)
z_moe      = LayerNorm(Dropout(out) + z_G)
```

Each expert is `Linear(D, D) → GELU → Dropout → Linear(D, D)`.

#### 4. Multi-Head Self-Attention (MHA)

Temporal self-attention over the T timesteps per stock:

```
z_attn = MHA(z_moe, z_moe, z_moe)           [H_mha heads]
z_out  = LayerNorm(Dropout(z_attn) + z_moe)
```

#### Data flow summary

```
x (N,T,D) → BiGRU → Gate → LN → SparseMoE → LN → MHA → LN → Z_temp (N,T,D)
```

---

## Path A — Temporal-Causal Hypergraph (TCH)

**Input**: `Z_temp (N, T, D)` — full temporal cube from MAGE
**Output**: `h_causal (N, D)` — causality-enriched final-timestep embedding

Discovers **asynchronous lead-lag** causal relationships: at time `t`, stock `i`
may lead stock `j` by `k` steps.

### Flow

```
Z_temp (N, T, D)
  → permute + reshape           : Z_flat (T·N, D)   [each row = one (time, stock) node]
  → Causal MHA with block mask                       [node at (t,s) only attends to t'≤t]
  → LayerNorm + residual
  → Linear(D,D) → GELU → Linear(D,M1)
  → ReTanh(·)                   : H_TCH (T·N, M1)   [sparse soft incidence matrix]
  → Hypergraph conv              : Z' = ELU(H·(Hᵀ·Proj(Z))) + Z   [O(T·N·M1·D)]
  → LayerNorm + residual
  → reshape (N, T, D) → slice [:, -1, :]            : h_causal (N, D)
```

### Causal attention mask

Prevents future-data leakage:

```
mask[i, j] = −∞   if t_i < t_j   (node i is earlier → cannot attend to later node j)
mask[i, j] =  0   otherwise

where t_i = i // N   (time index of flattened node i)
```

### ReTanh activation

```
ReTanh(x) = 0        if x ≤ 0
           = tanh(x) if x > 0
```

Combines ReLU sparsity (most hyperedge memberships are zero) with tanh
boundedness (prevents exploding incidence weights).

---

## Path B — Synchronous Relational Streams

All three sub-modules operate on `h_temp = Z_temp[:, -1, :]  (N, D)` — the
**pristine, unmixed** final-timestep state.

### Pos/Neg GAT

Multi-head additive graph attention on the explicit co-movement and
inverse-movement adjacency matrices from the data.

```
h_pos_raw, _ = GraphAttnMultiHead(D, d_gat, H_gat)(h_temp, pos_adj)   → (N, H_gat·d_gat)
h_pos         = Dropout(ReLU(Linear(H_gat·d_gat, D)(h_pos_raw)))       → (N, D)

h_neg_raw, _ = GraphAttnMultiHead(D, d_gat, H_gat)(h_temp, neg_adj)
h_neg         = Dropout(ReLU(Linear(H_gat·d_gat, D)(h_neg_raw)))       → (N, D)
```

#### Attention mechanism

```
support = h · W_attn                             (N, H·d)
f_1     = support · w_u                          (H, 1, N)
f_2     = support · w_v                          (H, N, 1)
logits  = LeakyReLU(f_1 + f_2) * adj_weight      (H, N, N)  [0-entries masked to −∞]
attn    = softmax(logits, dim=2)                             [nan→0 for isolated nodes]
output  = attn · support + residual(h)            (N, H·d)
```

### Global Probabilistic Hypergraph (GPH)

Discovers **latent macro-thematic** groupings (sector rallies, risk-off events,
factor rotations) beyond the explicit pairwise edges in pos/neg adj.

```
H_raw = Linear(D, M)(h_temp)                          (N, M)
H_GPH = col-softmax(ReTanh(H_raw))                    (N, M)   [prob dist over N per hyperedge]
w     = JSD_weights(H_GPH)                            (M,)     [uniqueness-based importance]
z     = ELU(H_GPH · diag(w) · (H_GPHᵀ · Proj(h))) + h
h_gph = LayerNorm(z)                                  (N, D)
```

#### JSD hyperedge weighting

Hyperedges that are more distinct from each other receive higher weight:

```
JSD(e_i, e_j) = 0.5·KL(e_i ‖ m) + 0.5·KL(e_j ‖ m),   m = 0.5·(e_i + e_j)
μ_i = mean_j JSD(e_i, e_j)             [average distinctiveness of edge i]
w   = softmax((μ − mean(μ)) / std(μ))  [normalised importance weights]
```

---

## 4-Stream Semantic Attention Fusion

Combines all four streams with a **per-stock** learned soft weighting:

```
all_streams = stack([h_causal, h_pos, h_neg, h_gph], dim=1)      (N, 4, D)

w    = Linear(D, 1)(Tanh(Linear(D, D)(all_streams)))              (N, 4, 1)
β    = softmax(w, dim=1)                                          (N, 4, 1)
fused = Σ_s  β_s · stream_s                                       (N, D)
```

Each stock independently weights the four views:
- `h_causal` — "which stocks sent causal shocks to me recently?"
- `h_pos`    — "what are my historically co-moving peers doing?"
- `h_neg`    — "what are my historically inverse peers doing?"
- `h_gph`    — "what macro theme am I currently grouped into?"

---

## Post-Fusion Normalisation & Prediction Head

```
fused = PairNorm-SI(fused)        [prevents over-smoothing after 4-stream fusion]
out   = Linear(D, 1)(fused)       (N, 1)   [raw return score]
```

**PairNorm-SI** (Scale-Individual mode):

```
x = x − col_mean(x)              [centering: remove market factor from fused reps]
x = x / ‖x_i‖_2                  [individual row normalisation]
```

---

## Loss Function

Three-term composite, defined in `train_hybrid.py → composite_loss`:

```
L = w_mse · MSE_cs  +  w_ic · L_IC  +  w_disp · L_disp
```

### MSE term (cross-sectional, normalised)

```
pred_cs   = pred − mean(pred)
target_cs = target − mean(target)
MSE_cs    = MSE(pred_cs, target_cs) / σ_return²        [σ_return = --return-scale = 0.02]
```

### IC term (differentiable soft Spearman)

```
rank(x)_i = Σ_j  σ((x_i − x_j) / τ)                  [soft rank via sigmoid]
soft_IC   = Pearson(rank(pred), rank(target).detach())
L_IC      = 1 − soft_IC
```

Temperature `τ` anneals: `τ = max(0.02,  0.2 × 0.95^epoch)`.
At epoch 1 τ ≈ 0.19 (smooth), at epoch 45+ τ = 0.02 (near exact Spearman).

### Dispersion penalty

```
ratio  = std(pred) / std(target)
L_disp = ReLU(r_min − ratio) + ReLU(ratio − r_max)     [r_min=0.3, r_max=3.0]
```

Prevents prediction collapse (all stocks scored identically).

### Default weights (current training run)

| Weight | Value | Role |
|--------|-------|------|
| `mse_weight` | 0.3 | Absolute accuracy |
| `ic_weight` | 1.2 | Cross-sectional ranking (primary objective) |
| `dispersion_weight` | 0.1 | Spread guard |

IC loss dominates by design: a model with poor ranking but low MSE is useless
for portfolio construction.

---

## Learning Rate Schedule

Three-phase schedule (defaults: warmup=5, plateau=20, total=150 epochs):

```
Phase 1 — Linear warmup   epochs  0– 4   LR: 0.1·η_max → η_max
Phase 2 — Flat plateau    epochs  5–24   LR: η_max  (constant)
Phase 3 — Cosine decay    epochs 25–149  LR: η_max → η_min

η_max = 2×10⁻⁴,   η_min = 1×10⁻⁵
```

The 20-epoch plateau keeps the LR at peak long enough for the model to exploit
full gradient signal before decay begins.

**LR at key epochs:**

| Epoch | LR |
|-------|----|
| 1 | 5.6×10⁻⁵ |
| 5 | 2.0×10⁻⁴ (peak) |
| 25 | 2.0×10⁻⁴ (still peak) |
| 65 | 1.6×10⁻⁴ (78% of peak) |
| 100 | 7.6×10⁻⁵ |
| 149 | 1.0×10⁻⁵ (floor) |

---

## Early Stopping

| Criterion | Default | Behaviour |
|-----------|---------|-----------|
| IC patience | 20 epochs | Stops if test Rank-IC does not improve for 20 consecutive epochs |
| Divergence guard | ratio > 3.5 for 5 epochs | Stops if `test_loss / train_loss` diverges |

Checkpoint saved whenever `test_rank_ic` strictly improves, or ties on IC with lower MSE.

---

## Design Decisions — What Was Kept, Dropped, Changed

| Component | Decision | Reason |
|-----------|----------|--------|
| TCH | **Reinstated** | Captures asynchronous cross-stock lead-lag causal chains that MAGE's per-stock MHA misses |
| MAGE (BiGRU-lite) | **Kept** | Per-stock temporal encoding with regime-specialised MoE; BiGRU replaces Mamba SSM to eliminate C++ dependency |
| Pos/Neg GAT | **Kept** | Explicit pre-computed correlation inductive bias complements learned hyperedges |
| GPH | **Kept** | Latent thematic groupings beyond pairwise correlations |
| 2D Feature Attention | **Dropped** | MAGE's MHA already captures cross-timestep feature deps per stock |
| IC-ranked composite loss | **Kept** (weight raised) | Directly optimises the ranking metric used in portfolio construction |
| Self-MLP stream | **Replaced by TCH** | Plain MLP adds no relational information |
| Cosine-to-zero LR | **Replaced** | 2-phase schedule decayed LR to near-zero by epoch 65 — model starved. 3-phase schedule keeps LR meaningful until epoch ~120 |

---

## Approximate Parameter Count

With default hyperparameters (D=64, L=1, E=4):

| Module | Approx. params |
|--------|----------------|
| Feature embedding (Linear F→D) | ~800 |
| MAGEBlock: BiGRU + Gate + MoE×4 + MHA | ~105 K |
| TCH: Causal MHA + FFN + Proj | ~42 K |
| Pos GAT + Neg GAT + projection MLPs | ~50 K |
| GPH: Linear + Proj | ~8 K |
| Semantic attention + PairNorm + Predictor | ~16 K |
| **Total** | **~222 K** |

---

## Usage

```bash
# Default training (2015–2024 train, 2025+ test):
python train_hybrid.py

# Reproduce the published run (2015–2023 train, 2024+ test):
python train_hybrid.py \
  --train-start-date 2015-01-01 --train-end-date 2023-12-31 \
  --test-start-date  2024-01-01 --test-end-date  2026-02-28

# Resume from a saved checkpoint:
python train_hybrid.py \
  --train-start-date 2015-01-01 --train-end-date 2023-12-31 \
  --test-start-date  2024-01-01 --test-end-date  2026-02-28 \
  --resume ../THGNN/data/model_saved/2023-12-29_hybrid_best.dat

# Larger model (N≥300 universe):
python train_hybrid.py --embed-dim 128 --num-hyper-edges 64 --num-tch-hyper-edges 64

# Run backtest on the trained checkpoint:
python backtest_hybrid.py --start-date 2024-01-01 --end-date 2026-02-28
```
