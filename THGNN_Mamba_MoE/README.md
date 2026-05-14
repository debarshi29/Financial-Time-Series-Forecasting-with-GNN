THGNN_Mamba_MoE
================

Sibling experiment to `THGNN_MaGNet` that keeps the same THGNN data paths,
training loop, backtest flow, TCH/GAT/GPH streams, and composite ranking loss,
but replaces the BiGRU temporal mixer with a Mamba+MoE block.

Run from this directory:

```bash
python train_hybrid.py
python backtest_hybrid.py --start-date 2024-01-01 --end-date 2026-02-28
```

Artifacts are saved with `_mamba_moe_*` names under the existing `../THGNN/data`
folders, so they do not overwrite the `_hybrid_*` BiGRU+MoE runs.
