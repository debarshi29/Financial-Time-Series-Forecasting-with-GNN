# Financial Time Series Forecasting with GNN

Multi-agent stock return forecasting system combining graph neural networks with live news sentiment.

## Project Structure

```
.
├── THGNN/                  # Standalone THGNN model (GRU + Pos/Neg GAT)
├── THGNN_MaGNet/           # Hybrid THGNN × MaGNet model (adds TCH + GPH + MAGE)
├── MultiAgent/             # Orchestrated 3-agent pipeline (GNN + News + Portfolio)
├── demo/                   # Streamlit web demo
└── Papers/                 # Reference papers
```

---

## Prerequisites

Each module has its own environment. Pick the one you need.

### THGNN environment

```bash
cd THGNN
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install torch numpy pandas matplotlib scikit-learn tqdm scipy
```

### THGNN_MaGNet environment

```bash
cd THGNN_MaGNet
pip install uv
uv sync                         # installs from pyproject.toml (CUDA 12.4 torch)
```

### MultiAgent + Demo environment

```bash
pip install -r MultiAgent/requirements.txt
# requires torch already installed (from either env above)
```

---

## 1. THGNN — Standalone Model

All commands run from the `THGNN/` directory with its venv active.

### Data pipeline (first-time setup)

```bash
python utils/download_market_data.py --start 2015-01-01
python utils/generate_relation.py --data-path data/nifty50.pkl --relation-dir data/relation
python utils/generate_data.py
```

### Train

```bash
# Standard IC-ranked training (recommended)
python train_ic_ranked.py \
  --train-start-date 2015-01-01 --train-end-date 2023-12-31 \
  --test-start-date 2024-01-01 --test-end-date 2026-02-28 \
  --hidden-dim 128 --num-heads 4 --epochs 80 --patience 20

# Walk-forward (4 expanding folds — runs train_ic_ranked.py internally)
python walk_forward_train.py --epochs 60 --patience 15

# Ablation: conservative / balanced / aggressive IC weighting
python run_icrank_tuning.py
```

Checkpoints are saved to `data/model_saved/` as `<date>_icrank_best.dat`.

### Backtest

```bash
python backtest.py \
  --start-date 2025-01-01 --end-date 2026-02-28 \
  --top-k 5
# Uses newest *_icrank_best.dat automatically.
# Outputs: data/backtest_results/<timestamp>/metrics_report.txt + metrics.json + equity_curve.png
```

To use a specific checkpoint:
```bash
python backtest.py --checkpoint data/model_saved/2024-12-31_icrank_best.dat
```

### Live predictions plot

```bash
python plot_live_predictions.py --start-date 2026-04-01 --end-date 2026-04-10 --top-n 5
# Saves plots to data/plots/
```

---

## 2. THGNN_MaGNet — Hybrid Model

All commands run from the `THGNN_MaGNet/` directory.

### Train (recommended on RTX 3080 via SSH)

```bash
python train_hybrid.py \
  --embed-dim 64 \
  --train-start-date 2015-01-01 --train-end-date 2023-12-31 \
  --test-start-date 2024-01-01 --test-end-date 2026-02-28 \
  --epochs 80 --lr 1e-4 --patience 15 --ic-warmup-epochs 5
```

Key hyperparameters:

| Arg | Default | Notes |
|-----|---------|-------|
| `--embed-dim` | 32 | 64 recommended on ≥12 GB GPU |
| `--epochs` | 60 | 80 gives more headroom |
| `--patience` | 15 | Early stop on test Spearman IC |
| `--ic-warmup-epochs` | 5 | IC loss ramps 0→full over 5 epochs |
| `--mse-weight` | 1.0 | Dominant loss term |
| `--ic-weight` | 0.2 | Spearman IC regularisation |

Checkpoint saved as `THGNN/data/model_saved/<date>_hybrid_best.dat`.

### Transfer checkpoint to local machine

```bash
scp user@remote:/path/to/THGNN/data/model_saved/*hybrid_best.dat \
    "THGNN/data/model_saved/"
```

### Live predictions plot

```bash
python plot_live_predictions.py --start-date 2026-04-01 --end-date 2026-04-10 --top-n 5
```

---

## 3. MultiAgent — Orchestrated Pipeline (LangGraph)

Three agents connected as a **LangGraph StateGraph**:

```
START → gnn_node → news_node → portfolio_node → END
                ↘  (--no-news)  ↗
```

All free: yfinance requires no API key; FinBERT runs locally (~420 MB, downloaded once);
LangGraph is open-source with no paid tier required.

### Install dependencies

```bash
pip install -r MultiAgent/requirements.txt
```

### Run

```bash
cd MultiAgent

# Full pipeline (GNN + news sentiment, via LangGraph)
python run_pipeline.py --date 2026-05-09 --top-k 10 --alpha 0.7

# GNN-only (skips news_node entirely via conditional edge)
python run_pipeline.py --date 2026-05-09 --no-news

# Save output to CSV
python run_pipeline.py --date 2026-05-09 --output portfolio_2026-05-09.csv

# Print the Mermaid diagram for the graph
python run_pipeline.py --print-graph
```

Arguments:

| Arg | Default | Description |
|-----|---------|-------------|
| `--date` | today | Trading date to predict for |
| `--top-k` | 10 | BUY/SELL candidates to highlight |
| `--alpha` | 0.7 | GNN signal weight (0=news-only, 1=GNN-only) |
| `--no-news` | False | Skip news agent (much faster) |
| `--checkpoint` | auto | Explicit path to `.dat` checkpoint |

**First run downloads FinBERT (~420 MB) to HuggingFace cache.**  
**Subsequent runs load from cache instantly.**

### How the agents work

**Agent 1 — GNN Prediction:** Loads the hybrid THGNN×MaGNet checkpoint, runs inference on the graph `.pkl` for the given date, outputs a raw predicted return score per stock.

**Agent 2 — News Sentiment:** Fetches recent headlines for each ticker via `yfinance.Ticker(symbol).news` (free, no key). Scores each headline with `ProsusAI/finbert` (FinBERT). Aggregates: `sentiment_score = mean(positive) − mean(negative)`. Results are SQLite-cached for 1 hour.

**Agent 3 — Portfolio Constructor:** Cross-sectionally normalises both signals, fuses them as `final_score = alpha × gnn_rank + (1−alpha) × sentiment_norm`, ranks all stocks, and labels top-K as BUY, bottom-K as SELL.

---

## 4. Streamlit Demo

```bash
cd demo
streamlit run app.py
```

Opens at `http://localhost:8501` with four tabs:

| Tab | Contents |
|-----|----------|
| **Live Portfolio** | Run the pipeline for any date; interactive buy/sell table + scatter plot |
| **Model Comparison** | Backtest metrics table (Sharpe, IC, return) for all saved runs |
| **News Drill-Down** | Pick any Nifty500 stock; see raw headlines + FinBERT scores |
| **System Overview** | Architecture diagram + walk-forward results |

The demo uses `@st.cache_resource` so the GNN model and FinBERT load once and stay in memory across reruns.

---

## Available Checkpoints

All in `THGNN/data/model_saved/`:

| File | Type | Train end | Notes |
|------|------|-----------|-------|
| `2024-12-31_icrank_best.dat` | THGNN | Dec 2024 | Best standalone THGNN |
| `2025-06-30_icrank_best.dat` | THGNN | Jun 2025 | Walk-forward fold 4 |
| `2024-12-31_hybrid_best.dat` | Hybrid | Dec 2024 | Best hybrid (embed_dim=64) |
| `2023-12-29_hybrid_best.dat` | Hybrid | Dec 2023 | Earlier hybrid run |

---

## Reproducing All Results

```bash
# 1. THGNN backtest
cd THGNN
python backtest.py --start-date 2025-01-01 --end-date 2026-02-28 --top-k 5

# 2. THGNN walk-forward IC evaluation
python walk_forward_train.py --eval-only

# 3. Hybrid model backtest (after training on remote GPU)
python backtest.py \
  --checkpoint data/model_saved/2024-12-31_hybrid_best.dat \
  --start-date 2024-01-01 --end-date 2026-02-28 --top-k 5

# 4. Multi-agent pipeline test
cd ../MultiAgent
python run_pipeline.py --date 2026-05-09 --top-k 10

# 5. Launch demo
cd ../demo
streamlit run app.py
```
