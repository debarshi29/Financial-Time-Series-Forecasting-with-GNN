<div align="center">

# Graph-Structured Deep Learning for Indian Stock Return Prediction

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torch--geometric](https://img.shields.io/badge/PyG-2.x-3C8EE8)](https://pyg.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-4CAF50)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-F7DC6F)](THGNN/LICENSE)
[![Thesis](https://img.shields.io/badge/Thesis-PDF-DC143C?logo=adobeacrobatreader&logoColor=white)](Report/BDA_Debarshi_Graph-Structured%20Deep%20Learning.pdf)

**MSc Big Data Analytics — Debarshi Chakraborty — 2026**

*A complete research framework for cross-sectional stock return forecasting on the NIFTY 500 universe, combining Graph Neural Networks with a six-agent LangGraph decision-support pipeline.*

</div>

---

## Overview

This repository contains the full implementation of an MSc thesis investigating graph-structured deep learning for Indian equity markets. The project makes three distinct contributions:

1. **Algorithmic** — A novel hybrid architecture extending THGNN with a MaGNet temporal encoder (BiGRU + Sparse MoE + Multi-Head Attention) and a dual hypergraph relational framework (Temporal-Causal Hypergraph + Global Probabilistic Hypergraph), applied to the NIFTY 500 universe.

2. **Empirical** — A rigorously controlled baseline THGNN implementation on NIFTY 500 providing the reference against which the hybrid is evaluated, including a full data leakage audit and walk-forward validation protocol.

3. **Systems** — A six-agent LangGraph pipeline integrating GNN-based stock scoring, FinBERT news sentiment, macro regime detection, risk profiling, and AI-generated research notes — deployed as an interactive trading-terminal web application.

---

## Results

### Backtest Performance (Jan 2024 – Feb 2026, NIFTY 500, Top-5 Long/Short)

| Metric | THGNN Baseline | THGNN × MaGNet (Hybrid) | THGNN Mamba-MoE |
|---|:---:|:---:|:---:|
| Spearman IC (mean) | 0.028 | **0.041** | 0.035 |
| Directional Accuracy | 51.3% | **53.2%** | 52.5% |
| Sharpe Ratio | — | **> 1.0** | — |
| Max Drawdown | — | — | — |

> Full equity curves, rolling IC, and quintile return charts are in `THGNN/data/backtest_results/` and `comparison_results/`.

### Sample Portfolio Output

```
Date: 2026-05-27  ·  Market Regime: BULL  ·  NIFTY 50 +1.32%  ·  VIX 16.7

TOP BUY CANDIDATES                          TOP SELL CANDIDATES
─────────────────────────────────────       ────────────────────────────────────
 JUBLPHARMA.NS   score=1.000  risk=LOW       INDIGO.NS    score=0.446  risk=HIGH
 EIDPARRY.NS     score=0.900  risk=LOW       DIVISLAB.NS  score=0.420  risk=LOW
 DEEPAKNTR.NS    score=0.869  sent=+0.931    MCDOWELL-N   score=0.389  risk=MED
 MGL.NS          score=0.811  sent=+0.937
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Six-Agent LangGraph Pipeline                           │
│                                                                                 │
│  ┌───────────┐   ┌──────────┐   ┌───────────┐   ┌─────────┐   ┌────────────┐  │
│  │ GNN Agent │──▶│ Macro    │──▶│ News Agent│──▶│ Risk    │──▶│ Portfolio  │  │
│  │           │   │ Agent    │   │           │   │ Agent   │   │ Agent      │  │
│  │ Hybrid    │   │ VIX+     │   │ yfinance+ │   │ Vol,DD, │   │ Alpha-     │  │
│  │ THGNN×    │   │ Regime   │   │ FinBERT   │   │ ATR,    │   │ blend +    │  │
│  │ MaGNet    │   │ detect.  │   │ (cached)  │   │ 52w rng │   │ BUY/SELL   │  │
│  └───────────┘   └──────────┘   └───────────┘   └─────────┘   └────────────┘  │
│        ↑                                                             │          │
│        │                                                             ▼          │
│  ┌─────────────────────────────────┐                    ┌────────────────────┐ │
│  │         Hybrid Model            │                    │   Report Agent     │ │
│  │                                 │                    │ AI research note   │ │
│  │  Input (N×T×4 OHLC)             │                    │ (Gemini/local LLM) │ │
│  │      │                          │                    └────────────────────┘ │
│  │  Linear Embed → MaGNet Encoder  │                                           │
│  │      (BiGRU + MoE + Attention)  │                                           │
│  │      │                          │                                           │
│  │  ┌───┴───────────┐              │                                           │
│  │  │ TCH Path      │ Pos/Neg GAT  │                                           │
│  │  │ (Temporal     │ + GPH Path   │                                           │
│  │  │  Causal HG)   │ (Co-movement)│                                           │
│  │  └───┬───────────┘              │                                           │
│  │      │                          │                                           │
│  │  Semantic Attention Fusion       │                                           │
│  │  PairNorm-SI → Linear → (N×1)   │                                           │
│  └─────────────────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
.
├── THGNN/                          # Baseline model — GRU + Heterogeneous GAT
│   ├── model/Thgnn.py              # StockHeteGAT architecture
│   ├── trainer/trainer.py          # Training helpers
│   ├── utils/
│   │   ├── download_market_data.py # Step 1 — OHLCV data via yfinance
│   │   ├── generate_relation.py    # Step 2 — Per-day Pearson correlation matrices
│   │   ├── generate_data.py        # Step 3 — Assemble graph samples (.pkl)
│   │   └── filter_nifty500.py      # NSE universe filtering
│   ├── train_ic_ranked.py          # PRIMARY training (IC-ranked composite loss)
│   ├── train_paper_bce.py          # Ablation — original BCE loss
│   ├── walk_forward_train.py       # 4-fold expanding-window walk-forward
│   ├── run_icrank_tuning.py        # Ablation — conservative/balanced/aggressive IC
│   ├── backtest.py                 # Historical evaluation with transaction costs
│   ├── plot_live_predictions.py    # Inference visualisation for a date range
│   ├── rebuild_graph_data.py       # Rebuild graphs with a different adj threshold
│   ├── LEAKAGE_ANALYSIS_REPORT.md  # Data leakage audit (forward-look prevention)
│   └── THESIS_MODEL_PIPELINE_DOCUMENTATION.md
│
├── THGNN_MaGNet/                   # Hybrid model — main thesis contribution
│   ├── model/hybrid_model.py       # Full hybrid architecture
│   ├── train_hybrid.py             # IC-ranked training with MaGNet warmup
│   ├── backtest_hybrid.py          # Hybrid backtest evaluation
│   ├── plot_live_predictions.py    # Live prediction charts
│   └── ARCHITECTURE.md             # Detailed design documentation
│
├── THGNN_Mamba_MoE/                # Experimental — Mamba SSM replaces BiGRU
│   ├── model/hybrid_model.py       # Mamba-based temporal encoder
│   ├── train_hybrid.py
│   ├── backtest_hybrid.py
│   └── ARCHITECTURE.md
│
├── MultiAgent/                     # Six-agent LangGraph orchestration pipeline
│   ├── graph.py                    # StateGraph definition
│   ├── orchestrator.py             # Pipeline coordinator
│   ├── run_pipeline.py             # CLI entry point
│   ├── agents/
│   │   ├── gnn_agent.py            # GNN inference node
│   │   ├── news_agent.py           # FinBERT sentiment node
│   │   ├── macro_agent.py          # Macro context (VIX, market regime)
│   │   ├── risk_agent.py           # Per-stock risk computation
│   │   ├── portfolio_agent.py      # Signal fusion + ranking
│   │   └── report_agent.py         # AI-generated executive summary
│   └── utils/
│       ├── finbert_loader.py       # HuggingFace FinBERT integration
│       └── news_cache.py           # SQLite sentiment cache (1-hour TTL)
│
├── demo/                           # Dark trading-terminal web dashboard
│   ├── server.py                   # Lightweight Python HTTP server
│   ├── api.py                      # REST API endpoints
│   └── templates/index.html        # Single-page trading terminal UI
│
├── frontend/                       # Vite + React SPA (alternative UI)
│   └── src/
│
├── Report/                         # LaTeX thesis source + compiled PDF
│   ├── main.tex
│   ├── reference.bib
│   ├── figures/
│   └── BDA_Debarshi_Graph-Structured Deep Learning.pdf
│
├── Papers/                         # Reference literature (16 PDFs)
├── comparison_results/             # Model comparison charts
├── stock_analysis_results/         # Stock ranking analysis charts
├── compare_models.py               # Multi-model backtest comparison
├── stock_analysis.py               # Comprehensive stock ranking analysis
├── alpha_sweep.py                  # GNN/sentiment fusion weight sweep
├── Dockerfile
└── pyproject.toml
```

---

## Setup

Each major component has its own isolated environment. Install only what you need.

### Baseline Model — THGNN

```bash
cd THGNN
pip install -r requirements.txt
```

### Hybrid Model — THGNN_MaGNet / THGNN_Mamba_MoE

Requires CUDA 12.4 (RTX 30/40 series recommended). Uses [uv](https://docs.astral.sh/uv/) for reproducible installs.

```bash
pip install uv
cd THGNN_MaGNet          # or THGNN_Mamba_MoE
uv sync
```

### Multi-Agent Pipeline + Demo

```bash
pip install -r MultiAgent/requirements.txt
```

### Environment Variables

Create a `.env` file at the repo root (only needed for LLM-enhanced reports):

```bash
GOOGLE_API_KEY=your_gemini_api_key   # Optional — for AI-generated research notes
```

FinBERT runs locally (no API key). yfinance requires no key.

---

## Usage

### 1 · THGNN — Baseline Model

All commands run from `THGNN/` with its environment active.

#### Data Pipeline (first-time setup)

```bash
# Step 1: Download 10+ years of OHLCV data for NIFTY 50/500
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
# IC-ranked composite loss (recommended)
python train_ic_ranked.py \
  --train-start-date 2015-01-01 --train-end-date 2023-12-31 \
  --test-start-date  2024-01-01 --test-end-date  2026-02-28 \
  --hidden-dim 128 --num-heads 4 --epochs 80 --patience 20

# 4-fold expanding-window walk-forward validation
python walk_forward_train.py --epochs 60 --patience 15

# IC-weight ablation study
python run_icrank_tuning.py
```

Checkpoints are saved to `data/model_saved/<date>_icrank_best.dat`.

#### Backtest & Inference

```bash
python backtest.py --start-date 2025-01-01 --end-date 2026-02-28 --top-k 5

python plot_live_predictions.py --start-date 2026-04-01 --end-date 2026-04-10 --top-n 5
```

---

### 2 · THGNN_MaGNet — Hybrid Model

All commands run from `THGNN_MaGNet/` with the uv environment active.

#### Training

```bash
python train_hybrid.py \
  --embed-dim 64 \
  --train-start-date 2015-01-01 --train-end-date 2023-12-31 \
  --test-start-date  2024-01-01 --test-end-date  2026-02-28 \
  --epochs 80 --lr 1e-4 --patience 15 --ic-warmup-epochs 5
```

| Argument | Default | Notes |
|---|---|---|
| `--embed-dim` | `32` | `64` recommended on ≥ 12 GB GPU |
| `--num-experts` | `4` | MoE expert count |
| `--ic-weight` | `0.2` | Spearman IC regularisation weight |
| `--dispersion-weight` | `0.1` | Spread-ratio penalty weight |
| `--patience` | `15` | Early stop on validation Spearman IC |

Checkpoint → `THGNN/data/model_saved/<date>_hybrid_best.dat`

#### Backtest

```bash
python backtest_hybrid.py \
  --checkpoint ../THGNN/data/model_saved/2024-12-31_hybrid_best.dat \
  --start-date 2024-01-01 --end-date 2026-02-28 --top-k 5
```

---

### 3 · Multi-Agent Pipeline

```bash
cd MultiAgent

# Full pipeline: GNN + macro regime + FinBERT sentiment + risk + report
python run_pipeline.py --date 2026-05-27 --top-k 10 --alpha 0.7

# GNN-only mode (no news download, fastest)
python run_pipeline.py --date 2026-05-27 --no-news

# Save portfolio to CSV
python run_pipeline.py --date 2026-05-27 --output portfolio_2026-05-27.csv

# Print the LangGraph DAG as Mermaid
python run_pipeline.py --print-graph
```

| Argument | Default | Description |
|---|---|---|
| `--date` | today | Trading date for inference |
| `--top-k` | `10` | Number of BUY/SELL candidates |
| `--alpha` | `0.7` | GNN weight (`0` = news only, `1` = GNN only) |
| `--no-news` | off | Skip the news agent |

> **First run:** FinBERT (~420 MB) downloads automatically to the HuggingFace cache. Subsequent runs are instant.

---

### 4 · Demo — Trading Terminal

```bash
cd demo
python server.py
# Opens at http://localhost:8000
```

| Tab | Contents |
|---|---|
| **Portfolio** | Run the pipeline for any date; interactive BUY/SELL table with scatter plot |
| **News** | Per-ticker headlines with FinBERT sentiment scores and article drill-down |
| **Overview** | Multi-agent architecture diagram + walk-forward validation summary |
| **Risk** | Per-stock volatility, ATR, 52-week range, drawdown metrics |
| **Report** | AI-generated daily research note (Markdown, copy-to-clipboard) |

---

### 5 · Analysis & Comparison Scripts

```bash
# Side-by-side model comparison (generates comparison_results/)
python compare_models.py

# Comprehensive per-stock alpha analysis (generates stock_analysis_results/)
python stock_analysis.py

# GNN/sentiment fusion weight sensitivity (alpha 0.0 → 1.0)
python alpha_sweep.py
```

---

## Model Checkpoints

All checkpoints are stored in `THGNN/data/model_saved/`:

| File | Type | Train End | Notes |
|---|---|---|---|
| `2024-12-31_hybrid_best.dat` | Hybrid MaGNet | Dec 2024 | **Best overall** — embed_dim=64 |
| `2023-12-29_hybrid_best.dat` | Hybrid MaGNet | Dec 2023 | Earlier run for comparison |
| `2024-12-31_icrank_best.dat` | THGNN Baseline | Dec 2024 | Best standalone baseline |
| `2025-06-30_icrank_best.dat` | THGNN Baseline | Jun 2025 | Walk-forward fold 4 |
| `2023-12-29_mamba_moe_best.dat` | Mamba-MoE | Dec 2023 | Experimental variant |
| `2023-12-29_icrank_{conservative,balanced,aggressive}.dat` | THGNN | Dec 2023 | IC-weight ablations |

---

## Reproducing All Results

```bash
# 1. THGNN baseline backtest
cd THGNN && python backtest.py --start-date 2025-01-01 --end-date 2026-02-28 --top-k 5

# 2. Hybrid model backtest
cd ../THGNN_MaGNet && python backtest_hybrid.py \
  --checkpoint ../THGNN/data/model_saved/2024-12-31_hybrid_best.dat \
  --start-date 2024-01-01 --end-date 2026-02-28 --top-k 5

# 3. Multi-agent pipeline live run
cd ../MultiAgent && python run_pipeline.py --date 2026-05-27 --top-k 10

# 4. Launch interactive demo
cd ../demo && python server.py

# 5. Model comparison plots
cd .. && python compare_models.py && python stock_analysis.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch 2.x, PyTorch Geometric |
| GNN | Heterogeneous GAT (pos/neg), Hypergraph convolution |
| Temporal Encoder | GRU, BiGRU, Mamba SSM, Sparse MoE, Multi-Head Attention |
| Agent Orchestration | LangGraph (StateGraph), LangChain |
| NLP / Sentiment | HuggingFace `ProsusAI/finbert` |
| Market Data | `yfinance` (free, no API key required) |
| Backtesting | Custom vectorised engine (pandas + numpy) |
| Visualisation | Matplotlib, Plotly |
| Demo | Python HTTP server, vanilla JS, Plotly.js |
| Packaging | `uv`, `pyproject.toml` |
| Container | Docker |

---

## Citation

If you use this code in your research, please cite the foundational papers:

```bibtex
@inproceedings{Xiang2022THGNN,
  author    = {Xiang, Sheng and Cheng, Dawei and Shang, Chencheng and Zhang, Ying and Liang, Yuqi},
  title     = {Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction},
  booktitle = {Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  year      = {2022},
  pages     = {3584--3593},
  doi       = {10.1145/3511808.3557089}
}

@article{Tan2025MaGNet,
  author  = {Tan, Yuxuan and others},
  title   = {MaGNet: Bidirectional Mamba Graph Network for Stock Movement Prediction},
  journal = {arXiv preprint},
  year    = {2025}
}
```

---

## License

MIT — see [THGNN/LICENSE](THGNN/LICENSE)

---

<div align="center">
<sub>MSc Big Data Analytics · Thesis — Debarshi Chakraborty · 2026</sub><br>
<sub>
<a href="Report/BDA_Debarshi_Graph-Structured%20Deep%20Learning.pdf">Read the full thesis (PDF)</a>
</sub>
</div>
