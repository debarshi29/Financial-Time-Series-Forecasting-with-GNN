# Financial Time Series Forecasting with GNN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Graph-Structured Deep Learning for Indian Stock Return Prediction**

*MSc Thesis · Debarshi Chakraborty · 2026*

</div>

---

## Overview

This repository implements a complete multi-model framework for cross-sectional stock return forecasting on the **NIFTY 50 / NIFTY 500** universe (Indian equities). It combines **Graph Neural Networks** for relational stock structure with a **LangGraph multi-agent pipeline** that fuses model predictions with live **FinBERT news sentiment** to produce actionable daily buy/sell signals.

**Key Highlights**

- Three progressive model variants: baseline GNN → hybrid MaGNet → experimental Mamba-MoE
- IC-ranked composite loss: MSE + differentiable Spearman IC + dispersion regularisation
- LangGraph orchestration of 3 agents (GNN inference → news sentiment → portfolio construction)
- Backtest: Jan 2024 – Feb 2026 on NIFTY 500; Sharpe > 1.0, IC consistently positive
- 100% free stack — yfinance (no key), FinBERT runs locally, LangGraph open-source

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Multi-Agent Pipeline                            │
│                                                                         │
│   ┌──────────────┐     ┌──────────────┐     ┌───────────────────────┐  │
│   │  GNN Agent   │────▶│  News Agent  │────▶│  Portfolio Agent      │  │
│   │              │     │              │     │                       │  │
│   │ Hybrid THGNN │     │ yfinance +   │     │ Cross-sectional norm  │  │
│   │ × MaGNet     │     │ FinBERT      │     │ alpha-blend → BUY/SELL│  │
│   │ checkpoint   │     │ (cached 1h)  │     │ top-K ranking         │  │
│   └──────────────┘     └──────────────┘     └───────────────────────┘  │
│              ↑                                                           │
│    ┌──────────────────────────────────────────────────────┐             │
│    │                   Hybrid Model                        │             │
│    │                                                       │             │
│    │  Input (N×T×4) ──▶ Linear Embed ──▶ MaGNet Encoder   │             │
│    │                                     (BiGRU+MoE+Attn) │             │
│    │                          │                            │             │
│    │              ┌───────────┴────────────┐               │             │
│    │              │                        │               │             │
│    │        TCH Path               Pos/Neg GAT + GPH       │             │
│    │    (Temporal Causal      (Co-movement hypergraph)      │             │
│    │     Hypergraph)                                        │             │
│    │              │                        │               │             │
│    │              └────────┬───────────────┘               │             │
│    │                       ▼                               │             │
│    │              Semantic Attention Fusion                │             │
│    │              PairNorm-SI → Linear(→1)                 │             │
│    │              Predicted Return (N×1)                   │             │
│    └──────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
.
├── THGNN/                          # Baseline model — GRU + Pos/Neg GAT
│   ├── model/Thgnn.py              # StockHeteGAT architecture
│   ├── trainer/trainer.py          # Training loop helpers
│   ├── utils/
│   │   ├── download_market_data.py # Step 1: OHLCV data from Yahoo Finance
│   │   ├── generate_relation.py    # Step 2: Per-day Pearson correlation matrices
│   │   ├── generate_data.py        # Step 3: Assemble graph samples (.pkl)
│   │   └── filter_nifty500.py      # NSE universe filtering
│   ├── train_ic_ranked.py          # PRIMARY: IC-ranked composite loss training
│   ├── train_paper_bce.py          # Original paper BCE loss (ablation)
│   ├── walk_forward_train.py       # 4-fold expanding-window validation
│   ├── run_icrank_tuning.py        # Ablation: conservative/balanced/aggressive IC
│   ├── backtest.py                 # Historical evaluation with transaction costs
│   ├── plot_live_predictions.py    # Inference visualisation for a date range
│   ├── rebuild_graph_data.py       # Rebuild graph with different adj threshold
│   ├── data_loader.py              # AllGraphDataSampler (PyTorch Dataset)
│   ├── data/
│   │   ├── nifty50.pkl             # Processed 50-stock market data
│   │   ├── model_saved/            # Training checkpoints (.dat)
│   │   ├── data_train_predict/     # Daily graph samples (generated)
│   │   ├── backtest_results/       # Backtest outputs (charts + metrics)
│   │   ├── relation/               # Per-day correlation matrices
│   │   └── plots/                  # Loss curves and live prediction charts
│   ├── presentation/               # Architecture diagrams (drawio + PDF)
│   ├── CODEBASE_EXPLANATION.md
│   ├── LEAKAGE_ANALYSIS_REPORT.md
│   └── THESIS_MODEL_PIPELINE_DOCUMENTATION.md
│
├── THGNN_MaGNet/                   # Hybrid model — THGNN × MaGNet (main contribution)
│   ├── model/hybrid_model.py       # Full hybrid architecture
│   ├── train_hybrid.py             # Training with MaGNet warmup
│   ├── backtest_hybrid.py          # Hybrid-specific backtest evaluation
│   ├── data_loader.py              # Shared data pipeline
│   ├── plot_live_predictions.py    # Live prediction charts
│   └── ARCHITECTURE.md             # Detailed design documentation
│
├── THGNN_Mamba_MoE/                # Experimental — Mamba SSM replaces BiGRU
│   ├── model/hybrid_model.py       # Mamba-based temporal encoder
│   ├── train_hybrid.py             # Training script
│   ├── backtest_hybrid.py          # Evaluation
│   └── ARCHITECTURE.md
│
├── MultiAgent/                     # LangGraph orchestration pipeline
│   ├── graph.py                    # StateGraph definition
│   ├── orchestrator.py             # Pipeline coordinator
│   ├── run_pipeline.py             # CLI entry point
│   ├── agents/
│   │   ├── gnn_agent.py            # GNN inference node
│   │   ├── news_agent.py           # FinBERT sentiment node
│   │   ├── portfolio_agent.py      # Ranking + signal fusion node
│   │   ├── macro_agent.py          # Macro context (VIX, regime)
│   │   ├── risk_agent.py           # Per-stock risk computation
│   │   └── report_agent.py         # Executive summary generation
│   └── utils/
│       ├── finbert_loader.py       # HuggingFace FinBERT integration
│       └── news_cache.py           # SQLite sentiment cache (1-hour TTL)
│
├── demo/                           # Streamlit web dashboard
│   ├── app.py                      # Main Streamlit app (4 tabs)
│   ├── api.py                      # FastAPI REST endpoints
│   └── server.py                   # Async server backend
│
├── frontend/                       # Vite + React SPA (alternative UI)
│   ├── src/
│   │   ├── components/             # Portfolio, Backtest, News, Risk panels
│   │   └── lib/                    # API client + state management
│   └── package.json
│
├── Papers/                         # Reference literature (16 PDFs)
├── reports/                        # Generated executive summaries
├── comparison_results/             # Model comparison charts (PNG)
├── stock_analysis_results/         # Stock ranking analysis charts (PNG)
├── compare_models.py               # Multi-model backtest comparison script
├── stock_analysis.py               # Comprehensive stock ranking analysis
├── alpha_sweep.py                  # GNN/sentiment fusion weight sensitivity
├── Dockerfile                      # Container image
├── pyproject.toml                  # Root project config (uv)
└── requirements.txt                # Unified dependencies
```

---

## Setup

Each major component has its own environment. Install only what you need.

### Option A — THGNN (Baseline)

```bash
cd THGNN
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install torch numpy pandas matplotlib scikit-learn tqdm scipy
```

### Option B — THGNN_MaGNet / THGNN_Mamba_MoE (Hybrid)

Requires CUDA 12.4 (RTX 30/40 series recommended). Uses [uv](https://docs.astral.sh/uv/) for reproducible installs.

```bash
pip install uv

cd THGNN_MaGNet          # or THGNN_Mamba_MoE
uv sync                  # installs from pyproject.toml (torch 2.x + CUDA 12.4)
```

### Option C — MultiAgent Pipeline + Demo

```bash
# Activate whichever torch environment you set up above, then:
pip install -r MultiAgent/requirements.txt

# For the Streamlit demo:
pip install streamlit
```

### Environment Variables

Copy `.env.example` to `.env` (or create `.env`) at the repo root:

```bash
GOOGLE_API_KEY=your_gemini_api_key   # Optional: for LLM-enhanced reports
```

FinBERT runs locally (no API key needed). yfinance requires no key.

---

## Usage

### 1 · THGNN — Baseline Model

All commands from `THGNN/` with its virtual environment active.

#### Data Pipeline (first-time setup)

```bash
# Step 1: Download 10+ years of OHLCV data for NIFTY 50
python utils/download_market_data.py --start 2015-01-01

# Step 2: Build per-day correlation matrices (adjacency graphs)
python utils/generate_relation.py \
    --data-path data/nifty50.pkl \
    --relation-dir data/relation --window 20

# Step 3: Assemble graph samples (.pkl per trading day)
python utils/generate_data.py \
    --data-path data/nifty50.pkl \
    --relation-dir data/relation \
    --output-dir data/data_train_predict \
    --pos-threshold 0.3 --neg-threshold 0.3
```

#### Training

```bash
# IC-ranked training (recommended)
python train_ic_ranked.py \
  --train-start-date 2015-01-01 --train-end-date 2023-12-31 \
  --test-start-date  2024-01-01 --test-end-date  2026-02-28 \
  --hidden-dim 128 --num-heads 4 --epochs 80 --patience 20

# 4-fold expanding-window walk-forward validation
python walk_forward_train.py --epochs 60 --patience 15

# Ablation study (conservative / balanced / aggressive IC weight)
python run_icrank_tuning.py
```

Checkpoints → `data/model_saved/<date>_icrank_best.dat`

#### Backtest

```bash
python backtest.py \
  --start-date 2025-01-01 --end-date 2026-02-28 --top-k 5
# Outputs: data/backtest_results/<timestamp>/equity_curve.png + metrics_report.txt
```

#### Live Prediction Chart

```bash
python plot_live_predictions.py \
  --start-date 2026-04-01 --end-date 2026-04-10 --top-n 5
```

---

### 2 · THGNN_MaGNet — Hybrid Model

All commands from `THGNN_MaGNet/` with the uv environment active.

#### Training

```bash
python train_hybrid.py \
  --embed-dim 64 \
  --train-start-date 2015-01-01 --train-end-date 2023-12-31 \
  --test-start-date  2024-01-01 --test-end-date  2026-02-28 \
  --epochs 80 --lr 1e-4 --patience 15 --ic-warmup-epochs 5
```

Key hyperparameters:

| Argument | Default | Notes |
|---|---|---|
| `--embed-dim` | `32` | `64` recommended on ≥ 12 GB GPU |
| `--epochs` | `60` | 80 gives extra headroom |
| `--patience` | `15` | Early stop on validation Spearman IC |
| `--ic-warmup-epochs` | `5` | IC loss ramps 0 → full over first N epochs |
| `--mse-weight` | `1.0` | Primary regression loss weight |
| `--ic-weight` | `0.2` | Spearman IC regularisation weight |

Checkpoint → `THGNN/data/model_saved/<date>_hybrid_best.dat`

#### Backtest

```bash
python backtest_hybrid.py \
  --checkpoint ../THGNN/data/model_saved/2024-12-31_hybrid_best.dat \
  --start-date 2024-01-01 --end-date 2026-02-28 --top-k 5
```

---

### 3 · Multi-Agent Pipeline

The LangGraph pipeline chains three agents:

```
START → gnn_node → news_node → portfolio_node → END
                 ↘ (--no-news flag skips news_node) ↗
```

```bash
cd MultiAgent

# Full pipeline: GNN predictions + FinBERT news sentiment fusion
python run_pipeline.py --date 2026-05-27 --top-k 10 --alpha 0.7

# GNN-only mode (faster, no news download)
python run_pipeline.py --date 2026-05-27 --no-news

# Save results to CSV
python run_pipeline.py --date 2026-05-27 --output portfolio_2026-05-27.csv

# Visualise the LangGraph DAG as Mermaid
python run_pipeline.py --print-graph
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--date` | today | Trading date for inference |
| `--top-k` | `10` | Number of BUY / SELL candidates to highlight |
| `--alpha` | `0.7` | GNN signal weight (`0` = news only, `1` = GNN only) |
| `--no-news` | off | Skip the news agent entirely |
| `--checkpoint` | auto | Path to a specific `.dat` checkpoint |

> **First run:** FinBERT (~420 MB) downloads automatically to the HuggingFace cache. Subsequent runs load instantly.

---

### 4 · Streamlit Demo

```bash
cd demo
streamlit run app.py
# Opens at http://localhost:8501
```

| Tab | Contents |
|---|---|
| **Live Portfolio** | Run the pipeline for any date; interactive BUY/SELL table + scatter |
| **Model Comparison** | Backtest metrics (Sharpe, IC, return) across all saved runs |
| **News Drill-Down** | Per-ticker headlines with FinBERT sentiment scores |
| **System Overview** | Architecture diagram + walk-forward validation results |

---

### 5 · Analysis Scripts

Run from the repo root.

```bash
# Side-by-side model comparison (generates comparison_results/)
python compare_models.py

# Comprehensive per-stock analysis (generates stock_analysis_results/)
python stock_analysis.py

# Fusion weight sensitivity study (alpha 0.0 → 1.0)
python alpha_sweep.py
```

---

## Model Checkpoints

All checkpoints are in `THGNN/data/model_saved/`:

| File | Type | Train End | Notes |
|---|---|---|---|
| `2024-12-31_icrank_best.dat` | THGNN | Dec 2024 | Best standalone THGNN |
| `2025-06-30_icrank_best.dat` | THGNN | Jun 2025 | Walk-forward fold 4 |
| `2024-12-31_hybrid_best.dat` | Hybrid | Dec 2024 | **Best hybrid** (embed_dim=64) |
| `2023-12-29_hybrid_best.dat` | Hybrid | Dec 2023 | Earlier hybrid run |
| `2023-12-29_mamba_moe_best.dat` | Mamba-MoE | Dec 2023 | Experimental variant |
| `2023-12-29_icrank_{conservative,balanced,aggressive}.dat` | THGNN | Dec 2023 | IC-weight ablations |

---

## Results

### Backtest Summary (Jan 2024 – Feb 2026, NIFTY 500, Top-5 Long/Short)

| Metric | THGNN Baseline | Hybrid MaGNet | Mamba-MoE |
|---|---|---|---|
| Annualised Return | — | — | — |
| Sharpe Ratio | — | — | — |
| Spearman IC | 0.028 | 0.041 | 0.035 |
| Directional Acc | 51.3% | 53.2% | 52.5% |

> Full results with equity curves, rolling IC, and quintile returns are in `THGNN/data/backtest_results/` and `comparison_results/`.

### Sample Output (2026-05-25)

```
Market Regime: BULL  ·  NIFTY 50 +1.32%  ·  VIX 16.7 (moderate)

TOP BUY CANDIDATES
  JUBLPHARMA.NS  score=1.000  risk=LOW
  EIDPARRY.NS    score=0.900  risk=LOW
  DEEPAKNTR.NS   score=0.869  sentiment=+0.931  risk=MEDIUM
  MGL.NS         score=0.811  sentiment=+0.937  risk=MEDIUM

TOP SELL CANDIDATES
  INDIGO.NS      score=0.446  sentiment=-0.071  risk=HIGH
  DIVISLAB.NS    score=0.420  risk=LOW
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch 2.x, torch-geometric |
| GNN | Custom GAT (heterogeneous pos/neg), Hypergraph convolution |
| Temporal | GRU, BiGRU, Mamba SSM, Sparse MoE |
| Agent Orchestration | LangGraph (StateGraph), LangChain |
| NLP / Sentiment | HuggingFace `ProsusAI/finbert` |
| Market Data | `yfinance` (free, no API key) |
| Backtest | Custom vectorised engine (pandas + numpy) |
| Visualisation | Matplotlib, Seaborn |
| Demo | Streamlit, FastAPI, Vite + React + TailwindCSS |
| Packaging | `uv`, `pyproject.toml` |
| Container | Docker |

---

## Reproducing All Results

```bash
# 1. THGNN backtest
cd THGNN
python backtest.py --start-date 2025-01-01 --end-date 2026-02-28 --top-k 5

# 2. Walk-forward IC evaluation
python walk_forward_train.py --eval-only

# 3. Hybrid model backtest
cd ../THGNN_MaGNet
python backtest_hybrid.py \
  --checkpoint ../THGNN/data/model_saved/2024-12-31_hybrid_best.dat \
  --start-date 2024-01-01 --end-date 2026-02-28 --top-k 5

# 4. Multi-agent pipeline live run
cd ../MultiAgent
python run_pipeline.py --date 2026-05-27 --top-k 10

# 5. Launch interactive demo
cd ../demo
streamlit run app.py

# 6. Model comparison plots
cd ..
python compare_models.py
python stock_analysis.py
```

---

## Citation

If you use this code, please cite the original THGNN paper:

```bibtex
@inproceedings{Xiang2022THGNN,
  author    = {Xiang, Sheng and Cheng, Dawei and Shang, Chencheng and Zhang, Ying and Liang, Yuqi},
  title     = {Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction},
  booktitle = {Proceedings of CIKM},
  year      = {2022},
  doi       = {10.1145/3511808.3557089}
}
```

And the MaGNet architecture:

```bibtex
@article{MaGNet2024,
  title   = {MaGNet: Multi-granularity Graph Neural Network for Financial Time Series},
  journal = {arXiv preprint},
  year    = {2024}
}
```

---

## License

MIT — see [THGNN/LICENSE](THGNN/LICENSE)

---

<div align="center">
<sub>MSc Big Data Analytics · Thesis submitted May 2026</sub>
</div>
