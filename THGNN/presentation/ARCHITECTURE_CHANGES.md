# Architecture Changes: This Implementation vs. Original THGNN Paper

**Reference:** "Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction"  
*(Wu et al., 2022 — the paper this repo is based on)*

---

## Summary Table

| Component | Original THGNN Paper | This Implementation | Change Type |
|---|---|---|---|
| Temporal encoder | GRU | GRU (same) | — |
| Input normalisation | Not specified / BatchNorm | **LayerNorm per timestep** | ADDED |
| Cross-sectional demean | ✗ | **`input - input.mean(dim=0)`** | ADDED |
| GAT heads (spatial) | Positive + Negative hetero-GAT | Same, but + **MLP projections before stacking** | MODIFIED |
| Semantic attention | 3-way softmax | Same | — |
| Graph normalisation | Not specified | **PairNorm PN-SI** after semantic attention | ADDED |
| Loss function | MSE (or classification CE) | **MSE + soft Spearman IC + dispersion** | MAJOR CHANGE |
| Ranking objective | Implicit / regression | **Explicit: soft Spearman IC loss** | ADDED |
| Prediction collapse guard | ✗ | **Dispersion penalty (spread-ratio)** | ADDED |
| IC warmup schedule | ✗ | **IC weight ramps 0 → 0.35 over 10 epochs** | ADDED |
| Temperature annealing | ✗ | **τ: 0.19 → 0.02 over training** | ADDED |
| Multi-horizon training | t+1, t+2, t+3 simultaneously | **Single-horizon t+1 only** | CHANGED |
| Market | Chinese A-share | **Nifty 50 (Indian)** | ADAPTED |
| Input features (F) | Multiple (varies by paper) | **12: OHLCV%chg + turnover%chg + mom5/10/20 + RSI14 + vol20 + cs\_rank** | ADAPTED/EXTENDED |
| Flexible input size | ✗ | **`Thgnn_flexible.py` auto-resizes GRU** | ADDED |
| Residual connections in GAT | ✗ | **Linear projection skip-connection in each GAT head** | ADDED |
| Dropout placement | Minimal | **After GRU + after each MLP projection** | ADDED |

---

## Detailed Change Notes

### 1. Cross-Sectional Demeaning (ADDED)
```python
inputs = inputs - inputs.mean(dim=0, keepdim=True)
```
Applied **after** LayerNorm, **before** the GRU. Subtracts the market-wide average for each (timestep, feature) channel across all N=50 stocks. This removes the common market beta so the model focuses purely on stock-specific alpha — which is directly what the cross-sectional IC objective measures.

**Not in original paper.** Highly effective for ranking tasks.

---

### 2. LayerNorm per Timestep (ADDED)
```python
self.input_norm = nn.LayerNorm(in_features)   # applied to each (stock, timestep) vector
```
Prevents volume and turnover (which have much larger absolute magnitudes than return % changes) from dominating GRU gate activations.

**Original paper** may use batch normalisation or nothing.

---

### 3. MLP Projections Between GAT and Semantic Attention (MODIFIED)
```python
self.mlp_self = nn.Linear(hidden_dim, hidden_dim)
self.mlp_pos  = nn.Linear(out_features*num_heads, hidden_dim)  # 256 → 128
self.mlp_neg  = nn.Linear(out_features*num_heads, hidden_dim)  # 256 → 128
```
After each GAT head, a separate MLP + ReLU + Dropout projects the output back to `hidden_dim=128` before the three embeddings are stacked for semantic attention.

**Original paper** stacks GAT outputs directly.  
**Benefit:** (a) homogenises dimensions, (b) adds non-linearity to each channel, (c) allows the model to selectively filter which information passes to semantic weighting.

---

### 4. PairNorm PN-SI (ADDED)
```python
self.pn = PairNorm(mode='PN-SI')
```
Applied to the final embedding after semantic attention. Normalises each node's embedding individually (PN-SI = per-node scale), then subtracts the column mean. Prevents over-smoothing: without this, repeated GNN aggregation collapses all node embeddings toward the graph mean, destroying the stock-level differences needed for ranking.

**Not in original paper.**

---

### 5. Composite Loss: MSE + Soft Spearman IC + Dispersion (MAJOR CHANGE)

**Original:** Standard MSE or classification cross-entropy.

**This implementation:**
```
L = w_MSE * MSE(r̂, r) / σ₀²
  + w_IC  * (1 − soft_IC(r̂, r))
  + w_disp * L_spread
```

- **MSE term** — normalised by `σ₀=0.01` so it stays O(1).  
- **Soft Spearman IC** — differentiable ranking via pairwise sigmoid soft-ranks:  
  `r̃_i = Σ_{j≠i} σ((r̂_i − r̂_j)/τ) + 1`  
  Directly optimises for cross-sectional rank correlation (IC), the exact metric used for evaluation.  
- **Dispersion penalty** — penalises the model if prediction std-dev is outside `[0.2×σ_r, 2.0×σ_r]`. Prevents the model from collapsing to a constant output or being pathologically volatile.

---

### 6. IC Warmup (ADDED)
IC weight ramps linearly from 0 → 0.35 over the first 10 epochs. MSE fits first; then ranking pressure is gradually applied. Without this, the IC gradient dominates early training when predictions are noisy, destabilising convergence.

---

### 7. Temperature Annealing (ADDED)
```
τ_e = max(0.02, 0.2 × 0.95^e)
```
Large τ early → smooth gradients. Small τ later → near-discrete rank signal. The annealing schedule prevents the soft-rank from being too "flat" (uninformative) or too "sharp" (non-differentiable) throughout training.

---

### 8. Single-Horizon Training (CHANGED)
The original paper trains on multiple horizons (t+1, t+2, t+3) simultaneously.  
This implementation trains on **t+1 only**.

**Reason:** Empirically, `corr(r_{t+1}, r_{t+3}) ≈ −0.24`. Multi-horizon training introduces conflicting gradients — the model is penalised for short-term and long-term reversal simultaneously, which degrades t+1 IC. Using t+1 only aligns training exactly with the evaluation metric.

---

### 9. Residual Connections in Each GAT Head (ADDED)
```python
if self.residual:
    support = support + self.project(inputs)
```
Each `GraphAttnMultiHead` has an optional skip-connection that adds a linear projection of the input to the attention output. This stabilises training in deeper configurations and prevents vanishing gradients through the attention weights.

---

## Architecture Diagram — Key Differences

```
Original THGNN Paper:
  Input → GRU → [Pos-GAT, Self, Neg-GAT] → Stack → Semantic Attn → Predictor

This Implementation:
  Input → LayerNorm → CS-Demean → GRU → Dropout
        → Pos-GAT (+residual) → MLP+ReLU+Dropout
        → Self-MLP+ReLU+Dropout
        → Neg-GAT (+residual) → MLP+ReLU+Dropout
        → Stack (3 × 128) → Semantic Attn
        → PairNorm (PN-SI)
        → Linear Predictor
  
  Loss: MSE/σ₀² + IC_warmup·(1−soft_IC) + disp_weight·L_spread
  F = 12: open, high, low, close, to, vol (all %chg)
        + mom5, mom10, mom20 (cumulative return)
        + rsi14 (RSI-14/100)
        + vol20 (20-day realised vol)
        + cs_rank (cross-sectional percentile rank of close return, added in generate_data.py)
```
