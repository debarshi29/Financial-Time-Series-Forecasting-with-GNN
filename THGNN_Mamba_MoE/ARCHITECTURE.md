# Hybrid THGNN x MaGNet Mamba+MoE - Architecture Reference

This directory is a sibling variant of `THGNN_MaGNet`. It keeps the same data
loader, model input/output contract, graph streams, loss, training loop, and
backtesting flow, but swaps the temporal encoder from **BiGRU+MoE** to
**Mamba+MoE**.

## High-Level Flow

```text
Input (N, T, F)
  -> LayerNorm + cross-sectional demean
  -> Linear(F -> D)
  -> MAGEBlock x L
       Mamba/SelectiveSSM -> SparseMoE -> temporal MHA
  |
  +-> Path A: TemporalCausalHypergraph(Z_temp) -> h_causal
  |
  +-> Path B on h_temp = Z_temp[:, -1, :]
        +-> PosGAT(h_temp, pos_adj) -> h_pos
        +-> NegGAT(h_temp, neg_adj) -> h_neg
        +-> GPHypergraph(h_temp)    -> h_gph

4-stream SemanticFusion([h_causal, h_pos, h_neg, h_gph])
  -> PairNorm-SI
  -> Linear(D -> 1)
```

## What Changed From `THGNN_MaGNet`

Only the per-stock temporal mixer inside `MAGEBlock` changed:

```text
Old: x -> BiGRU -> forward/backward gate -> SparseMoE -> MHA
New: x -> Mamba/SelectiveSSM -> SparseMoE -> MHA
```

Everything after `Z_temp` is intentionally unchanged:

- TCH still consumes the full `(N, T, D)` temporal cube.
- Pos/Neg GAT and GPH still consume the final snapshot `(N, D)`.
- Semantic attention still fuses the same four streams.
- PairNorm, prediction head, composite IC-ranked loss, and date-split logic are unchanged.
- Default data path remains `../THGNN/data/data_train_predict`.

## Mamba Implementation

`model/hybrid_model.py` tries to import the official `mamba-ssm` package:

```python
from mamba_ssm import Mamba
```

If that import works, `MAGEBlock` uses the official Mamba layer. If it is not
installed, the code uses `SelectiveSSMFallback`, a pure-PyTorch Mamba-style
selective state-space mixer. This keeps the directory runnable on the current
Windows/PyTorch setup while still allowing a true `mamba-ssm` backend on a
compatible Linux/CUDA environment.

## Tensor Shapes

| Symbol | Meaning | Default |
|--------|---------|---------|
| N | Number of stocks in universe | dataset-dependent |
| T | Lookback timesteps per sample | dataset-dependent |
| F | Input features per stock per timestep | 12 |
| D | Core embedding dimension (`embed_dim`) | 64 |
| E | MoE experts (`num_moe_experts`) | 4 |
| H_mha | Temporal MHA heads (`num_mha_heads`) | 2 |
| H_gat | GAT heads (`gat_heads`) | 8 |
| M | GPH hyperedges (`num_hyper_edges`) | 32 |
| M1 | TCH causal hyperedges (`num_tch_hyper_edges`) | 32 |
| L | MAGE stack depth (`num_mage_layers`) | 1 |

## Artifact Names

This variant writes separate artifacts into the existing `THGNN/data` folders:

- Checkpoint: `*_mamba_moe_best.dat`
- Train log: `*_mamba_moe_train.txt`
- Metrics CSV: `*_mamba_moe_metrics.csv`
- Summary JSON: `*_mamba_moe_summary.json`
- Loss plot: `*_mamba_moe_loss_curve.png`

## Usage

Run from this directory:

```bash
python train_hybrid.py
```

Custom split:

```bash
python train_hybrid.py \
  --train-start-date 2015-01-01 --train-end-date 2023-12-31 \
  --test-start-date 2024-01-01 --test-end-date 2026-02-28
```

Backtest the most recent Mamba+MoE checkpoint:

```bash
python backtest_hybrid.py --start-date 2024-01-01 --end-date 2026-02-28
```

Use a specific checkpoint:

```bash
python backtest_hybrid.py --checkpoint ../THGNN/data/model_saved/2023-12-29_mamba_moe_best.dat
```
