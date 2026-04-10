# THGNN Thesis Technical Documentation

## Scope
This document describes the current implementation and experimental pipeline in:

- `model/Thgnn.py`
- `utils/generate_relation.py`
- `utils/generate_data.py`
- `main.py`
- `new_main.py`
- `train_ic_ranked.py`

Supporting behavior from:

- `trainer/trainer.py`
- `data_loader.py`

The goal is to provide thesis-ready technical details: data flow, architecture, loss functions, hyperparameters, training/evaluation logic, and key caveats.

---

## 1. End-to-End Pipeline

1. Build processed market dataframe (`data/nifty50.pkl`) from raw OHLCV.
2. Build relation matrices (`data/relation/*.csv`) using rolling-window correlation:
   - `utils/generate_relation.py`
3. Build graph samples (`data/data_train_predict/*.pkl`) and stock metadata (`data/daily_stock/*.csv`):
   - `utils/generate_data.py`
4. Train/evaluate:
   - Baseline training/prediction: `main.py`
   - Extended training + walk-forward + backtest: `new_main.py`
   - IC/dispersion-aware training: `train_ic_ranked.py`

---

## 2. Data Artifacts and Tensor Shapes

Each graph sample (`.pkl`) produced by `utils/generate_data.py` has:

- `features`: shape `(N, W, F)`  
  - `N`: number of stocks in relation matrix (typically 50)  
  - `W`: lookback window (`--window`, default 20)  
  - `F`: feature count (6)
- `labels`: shape `(N, H)`  
  - `H`: forecast horizon (`--horizon`, default 3)
- `pos_adj`: shape `(N, N)` positive-edge adjacency
- `neg_adj`: shape `(N, N)` negative-edge adjacency
- `mask`: length `N` boolean list

Feature columns:

- `open`, `high`, `low`, `close`, `to`, `vol`

Target semantics:

- `label` is next-step close return in decimal form (`pct_change`, not percentage points).
- Example: `0.01` means +1%.

---

## 3. Relation Construction (`utils/generate_relation.py`)

## 3.1 Inputs and Core Logic

- Input dataset: `data/nifty50.pkl` with `dt`, `code`, and engineered features.
- For each relation date `t`, script uses previous `window` trading days ending at `t`.
- For each stock pair, computes average Pearson correlation across six feature channels.

## 3.2 Correlation Computation

- `cal_pccs(...)`: manual Pearson formula.
- `calculate_pccs(...)`: computes pairwise per-feature correlations and averages across features.
- `stock_cor_matrix(...)`: generates full stock-by-stock relation matrix.

## 3.3 Leakage-Safe Date Behavior

Current behavior is daily relation generation:

- `infer_relation_dates(...)` returns all filtered trading dates (not monthly month-end only).
- This removes month-end lookahead leakage seen in older monthly assignment pipelines.

## 3.4 Hyperparameters and CLI

- `--window` default `20`
- `--processes` default `1`
- Optional date filters: `--start-date`, `--end-date`

Output:

- One CSV per relation date in `data/relation`, filename `YYYY-MM-DD.csv`.

---

## 4. Graph Sample Construction (`utils/generate_data.py`)

## 4.1 Logic

For each valid trading day `t` that has a relation file:

1. Load relation matrix for exactly day `t` via map `date -> relation_path`.
2. Build `pos_adj` and `neg_adj` using thresholds:
   - positive edges where correlation > `pos_threshold`
   - negative edges where correlation < `-neg_threshold`
3. For each stock in relation matrix:
   - take `W`-day feature window ending at `t`
   - take `H` labels from position `t` onward in stock timeline
4. Keep sample only if all relation stocks are fully aligned.

## 4.2 Adjacency Construction

- `pos_adj`: binary matrix from thresholded positive correlations.
- `neg_adj`: binary matrix from thresholded negative correlations.
- diagonals are removed.

## 4.3 Hyperparameters and CLI

- `--window` default `20`
- `--horizon` default `3`
- `--pos-threshold` default `0.1`
- `--neg-threshold` default `0.1`
- optional date range filters

Outputs:

- `data/data_train_predict/YYYY-MM-DD.pkl`
- `data/daily_stock/YYYY-MM-DD.csv` (contains `code`, `dt`)

---

## 5. Core Architecture (`model/Thgnn.py`)

The model class is `StockHeteGAT`.

## 5.1 Components

1. **Temporal encoder**: GRU
   - input size `in_features` (default 6)
   - hidden size `hidden_dim`
   - `num_layers`
   - `batch_first=True`

2. **Graph encoders**:
   - `pos_gat`: `GraphAttnMultiHead`
   - `neg_gat`: `GraphAttnMultiHead`

3. **Projection heads**:
   - `mlp_self`, `mlp_pos`, `mlp_neg` (all to `hidden_dim`)

4. **Semantic fusion**:
   - stack three embeddings `[self, pos, neg]`
   - `GraphAttnSemIndividual` computes attention over the 3 semantic channels

5. **Normalization + predictor**:
   - `PairNorm(mode='PN-SI')`
   - linear predictor to `predictor_out_dim` (horizon-aware)

## 5.2 Forward Pass (high level)

Given `(features, pos_adj, neg_adj)`:

1. `GRU(features) -> support`
2. `pos_gat(support, pos_adj)` and `neg_gat(support, neg_adj)`
3. project self/pos/neg embeddings
4. semantic attention fusion over 3 channels
5. PairNorm
6. linear prediction

## 5.3 Initialization

- Linear layers initialized with Xavier uniform (`gain=0.02`).

---

## 6. Baseline Training Script (`main.py`)

## 6.1 Purpose

- Standard train/val split by index.
- MSE-based regression training.
- Generates validation predictions.
- Performs walk-forward retraining and one-step inference.

## 6.2 Default Training Hyperparameters (`Args`)

- `device='cpu'`
- `max_epochs=60`
- `epochs_eval=10`
- `lr=0.0002`
- `scheduler=StepLR(step_size=5000, gamma=0.9)`
- `hidden_dim=128`
- `num_heads=8`
- `out_features=32`
- `batch_size=1`
- `loss_fcn=mse_loss`

## 6.3 Output Files

- Model checkpoints: `data/model_saved/{pre_data}_epoch_60.dat`
- Validation predictions: `data/prediction/pred.csv`
- Walk-forward predictions: `data/prediction/walk_forward_pred.csv`

---

## 7. Extended Script (`new_main.py`)

## 7.1 Purpose

- Enhanced operational script for:
  - robust data extraction (`extract_data_fixed`)
  - progress bars
  - training + prediction + walk-forward
  - portfolio backtesting and visualization

## 7.2 Key Differences vs `main.py`

- Auto device selection: CUDA if available.
- Additional error handling for malformed samples.
- Walk-forward inference gracefully skipped when no future sample exists.
- Backtesting module with:
  - total return, annualized return, volatility, Sharpe, Sortino, max drawdown, Calmar, win rate.

## 7.3 Backtesting Implementation Notes (thesis caveat)

- If true close is unavailable in prediction data, script attempts:
  1. merge from daily stock CSVs
  2. fallback synthetic close from cumulative product of return series in `nifty50.pkl`
- Synthetic close is suitable for relative/ranking strategy checks, but not exact absolute-price realism.
- Transaction-cost-heavy rebalance behavior can dominate outcomes if holdings are not mark-to-market with external execution assumptions.

## 7.4 Default `__main__` Split Behavior

- Trains up to `2023-12-29`, validates from `2024-01-01`.

---

## 8. IC-Ranked Training (`train_ic_ranked.py`)

## 8.1 Motivation

Baseline MSE training can collapse to near-mean predictions (low dispersion).  
`train_ic_ranked.py` adds ranking and dispersion pressure.

## 8.2 Composite Loss

For masked predictions `pred` and targets `target` (N stocks, one trading day):

- `mse = MSE(pred, target) / return_scale²`  — fixed-scale normalized MSE
- `corr = SoftSpearman(pred, target)` — differentiable cross-sectional rank IC (see below)
- `ic_loss = 1 - corr`
- `pred_std = std(pred)` — cross-sectional std of predictions for that day
- `target_std = max(std(target), return_scale × 0.1)` — cross-sectional std of targets, floored to prevent ratio explosion on low-dispersion days (when all stocks move in lockstep)
- `spread_ratio = pred_std / target_std` — dimensionless spread ratio
- `dispersion_penalty = ReLU(min_dispersion_ratio − spread_ratio) + ReLU(spread_ratio − max_dispersion_ratio)`

Final objective:

`loss = mse_weight * mse + ic_weight * ic_loss + dispersion_weight * dispersion_penalty`

**MSE normalization rationale:** Raw MSE on decimal daily returns has magnitude ~(0.01)² = 1e-4, whereas the IC loss is O(1). Without normalization, the MSE gradient is ~3500× smaller than the IC gradient, making the MSE term effectively dead. Dividing by the fixed constant `return_scale² = (0.01)² = 1e-4` brings MSE to O(1), so `mse_weight` and `ic_weight` have their intended relative effect. A fixed constant is used rather than per-batch `target_var`, which varied too much across days and between splits (causing val/test loss to spike on low-volatility days).

**Dispersion penalty rationale:** The previous raw-std form (`ReLU(min_std - pred_std)`) had magnitude O(0.01), making it negligible against O(1) MSE/IC terms. The dimensionless ratio form is O(1) and also penalizes *over-spreading* (`spread_ratio > max_dispersion_ratio`), preventing the model from over-amplifying prediction variance. The ratio gradient is proportional to `1/pred_std`, which becomes stronger as predictions collapse toward zero — exactly when correction is most needed.

### Soft Spearman IC

The IC term uses a **differentiable soft-rank Spearman correlation** rather than raw Pearson:

1. For each stock `i`, compute its soft rank:
   `rank_i = Σ_j sigmoid((pred_i − pred_j) / τ)`
2. Compute Pearson correlation between soft ranks of predictions and (detached) soft ranks of targets.

The temperature `τ` is **annealed** during training: `τ = max(0.02, 0.2 × 0.95^epoch)`, starting at ~0.19 (smooth gradients when predictions are far from correct) and decaying to 0.02 by epoch 60 (sharper rank signal near convergence). A fixed small τ risks vanishing gradients when predictions are clustered near zero.

This equals the Spearman rank correlation in the limit τ → 0 and is fully differentiable via autograd. It is more robust than Pearson IC, which can be dominated by a single stock with an extreme return magnitude. Target ranks are detached from the computation graph — gradients flow only through prediction ranks.

**IC warmup:** The IC weight is ramped linearly from 0 to `ic_weight` over the first `ic_warmup_epochs` (default 10) epochs. This allows MSE to stabilize before ranking pressure is introduced. Critically, all three splits (train, val, test) use the same ramped weight at each epoch so loss curves are directly comparable. Previously, val/test used the full IC weight while train used the ramped weight, creating a visible discontinuity in the loss plot.

**Note:** The IC value reported during training (logged as `ic`) reflects the soft Spearman approximation. The `rank_ic` metric uses exact non-differentiable Spearman ranks and is used for checkpoint selection. Post-hoc evaluation scripts compute true Spearman IC via `pandas.Series.corr(method="spearman")`.

## 8.3 Default Hyperparameters

- date split:
  - train start: `2015-01-01`
  - train end: `2023-12-29` (default; no hard ceiling — any date with pkl data is valid)
  - val start: `2024-01-01`
  - val end: `2024-12-31`
  - test start: `2025-01-01`
  - test end: `2026-02-28`
- model:
  - `hidden_dim=64`, `num_heads=4`, `num_layers=1`, `out_features=32`, `in_features=12`
- optimizer:
  - `AdamW(lr=1e-4, weight_decay=1e-4)`
  - LR schedule: 5-epoch linear warmup (0.1× → 1× LR), then `CosineAnnealingLR(T_max=epochs-5)`
- loss weights:
  - `mse_weight=1.0`
  - `ic_weight=0.35`
  - `dispersion_weight=0.2`
  - `min_dispersion_ratio=0.2`
  - `max_dispersion_ratio=2.0`
  - `return_scale=0.01` (decimal returns; adjust to `1.0` for percentage-point returns)
- IC objective: soft Spearman rank correlation with annealed temperature `τ = max(0.02, 0.2 × 0.95^epoch)`
- IC warmup: weight ramped from 0 → `ic_weight` over first `ic_warmup_epochs=10` epochs
- training:
  - `epochs=60`
  - early stop patience `20`
  - seed `42`

## 8.4 Model Selection

- Best checkpoint selected by:
  1. highest validation IC
  2. tie-break by lower validation MSE

Saved as:

- `data/model_saved/{pre_data}_icrank_best.dat`

Checkpoint includes:

- model weights
- best epoch
- `val_ic`, `val_mse`
- training config
- split indices

---

## 9. Training Utilities That Affect Results

## 9.1 Losses (`trainer/trainer.py`)

- `mse_loss`: main regression loss.
- `bce_loss`: binary classification loss mode.

## 9.2 Data Extraction

- `extract_data(...)` handles batch-dim squeezing conservatively.
- `train_epoch(...)` skips malformed samples and non-3D feature tensors.
- `eval_epoch(...)` averages loss across full validation loader.

## 9.3 Dataset Split and Purging (`data_loader.py`)

- `AllGraphDataSampler` uses index ranges:
  - train: `[data_start, data_middle)`
  - val: `[data_middle, data_end)`
- Supports `purge_gap`; if omitted, inferred as `horizon - 1` from labels.
- Invalid/malformed samples are skipped during loading.

---

## 10. Important Thesis Discussion Points

## 10.1 Leakage Control

- Relation generation and sample generation are now date-aligned per day.
- This is critical: graph at date `t` uses relation ending at `t`, not future month-end relations.

## 10.2 Why Low MSE May Be Misleading

- Targets are decimal returns with small magnitude (~0.01 std).
- The normalized MSE reported in `train_ic_ranked.py` is `MSE / return_scale²` (not raw MSE). A value of 1.0 means the model predicts no better than the cross-sectional mean. Values < 1.0 indicate genuine predictive accuracy.
- Raw MSE ≈ normalized_MSE × (0.01)² — small raw values do not imply strong directional skill.
- Must report:
  - IC (cross-sectional and/or per-stock)
  - sign/directional accuracy
  - prediction spread ratio (`pred_std / target_std`)
  - portfolio spread metrics

## 10.3 Calibration vs Ranking Tradeoff

- MSE-only models tend to under-dispersed predictions.
- IC/dispersion regularization increases ranking signal but can over-amplify volatility.
- Hyperparameter ablation (e.g., conservative/balanced/aggressive) is essential.

## 10.4 Backtest Interpretation Limits

- If synthetic close is used, PnL is approximate.
- Rebalancing assumptions and transaction cost model materially affect returns.
- Emphasize robustness checks (turnover sensitivity, slippage assumptions).

---

## 11. Suggested Thesis Tables/Figures

1. Data pipeline diagram:
   - raw OHLCV -> `nifty50.pkl` -> relation matrices -> graph samples.
2. Model architecture figure:
   - GRU + dual GAT + semantic attention + predictor.
3. Hyperparameter table:
   - baseline (`main.py`), extended (`new_main.py`), IC-ranked (`train_ic_ranked.py`).
4. Leakage-control comparison:
   - old monthly relation assignment vs current daily alignment.
5. Ablation table:
   - `(ic_weight, dispersion_weight, min_dispersion_ratio)` vs `(val IC, val MSE, spread ratio)`.
6. Diagnostic plots:
   - actual vs predicted return
   - prediction dispersion metrics
   - top/bottom portfolio spread over time.

---

## 12. Reproducibility Checklist

- Fix random seeds (`train_ic_ranked.py` sets seed).
- Document exact commit hash and dataset date range.
- Save all checkpoint configs and split metadata.
- Report GPU/CPU environment and PyTorch version.
- Report filtering rules for invalid samples.

---

## 13. Quick Reference: Default Outputs

- Relations: `data/relation/*.csv`
- Graph samples: `data/data_train_predict/*.pkl`
- Daily stock maps: `data/daily_stock/*.csv`
- Baseline predictions: `data/prediction/pred.csv`
- Walk-forward predictions: `data/prediction/walk_forward_pred.csv`
- Checkpoints:
  - baseline: `data/model_saved/{pre_data}_epoch_60.dat`
  - IC-ranked best: `data/model_saved/{pre_data}_icrank_best.dat`
- Backtest results: `data/backtest_results/*`

