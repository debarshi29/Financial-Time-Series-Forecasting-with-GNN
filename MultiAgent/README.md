# Multi-Agent LangGraph Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-4CAF50)](https://github.com/langchain-ai/langgraph)
[![FinBERT](https://img.shields.io/badge/FinBERT-ProsusAI-8A2BE2)](https://huggingface.co/ProsusAI/finbert)

A six-agent [LangGraph](https://github.com/langchain-ai/langgraph) orchestration pipeline that integrates GNN-based stock scoring, macro regime detection, FinBERT news sentiment, systematic risk profiling, and AI-generated research notes into a production-style daily decision-support system for the NIFTY 500 universe.

---

## Pipeline Overview

```
START
  │
  ▼
┌─────────────────┐
│   GNN Agent     │  Load THGNN × MaGNet checkpoint → run inference for date
│  gnn_agent.py   │  Output: per-stock predicted returns + normalised scores
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Macro Agent    │  Fetch NIFTY 50 index data + VIX via yfinance
│  macro_agent.py │  Detect market regime: BULL / BEAR / SIDEWAYS
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  News Agent     │  Fetch top-K ticker headlines via yfinance news
│  news_agent.py  │  Run FinBERT locally → per-stock sentiment score
│                 │  Results cached in SQLite (1-hour TTL)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Risk Agent     │  Compute: 20d volatility, ATR, 52-week range position,
│  risk_agent.py  │  max drawdown, Δ from high/low → risk tier (LOW/MED/HIGH)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Portfolio Agent │  Fuse GNN score + sentiment: final_score = α·gnn + (1-α)·sent
│portfolio_agent.py│ Rank cross-sectionally → assign BUY / HOLD / SELL
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Report Agent   │  Generate an executive Markdown research note
│  report_agent.py│  Uses Gemini (if GOOGLE_API_KEY set) or local LLM
└────────┬────────┘
         │
        END
```

---

## Quick Start

### Install

```bash
# From the repo root, with the THGNN_MaGNet/uv environment active:
pip install -r MultiAgent/requirements.txt
```

### Run

```bash
cd MultiAgent

# Full pipeline — GNN + macro + news + risk + portfolio + report
python run_pipeline.py --date 2026-05-27 --top-k 10 --alpha 0.7

# GNN-only (no network access, fastest)
python run_pipeline.py --date 2026-05-27 --no-news

# Save portfolio table to CSV
python run_pipeline.py --date 2026-05-27 --output portfolio_2026-05-27.csv

# Print the LangGraph DAG as a Mermaid diagram
python run_pipeline.py --print-graph
```

### Environment Variables

```bash
# .env (in MultiAgent/ or repo root)
GOOGLE_API_KEY=your_gemini_api_key   # Optional — enables AI-generated report
```

If `GOOGLE_API_KEY` is not set, the report agent falls back to a structured template summary.

---

## CLI Reference

### `run_pipeline.py`

| Argument | Default | Description |
|---|---|---|
| `--date` | today | Trading date for inference (YYYY-MM-DD) |
| `--top-k` | `10` | Number of top BUY and SELL candidates to highlight |
| `--alpha` | `0.7` | GNN signal weight in fusion (`0` = news only, `1` = GNN only) |
| `--no-news` | off | Skip the news agent; use GNN scores only |
| `--checkpoint` | auto | Path to a specific `.dat` model checkpoint |
| `--output` | — | If set, write portfolio table to this CSV path |
| `--print-graph` | off | Print the compiled LangGraph DAG as Mermaid and exit |

---

## Agent Reference

### `gnn_agent.py` — GNN Inference

Loads the hybrid THGNN × MaGNet checkpoint and runs forward inference on the graph built for the requested date. Returns a dict of `{ticker: predicted_return}` for all NIFTY 500 constituents present in the graph on that date, along with min-max normalised scores in `[0, 1]`.

### `macro_agent.py` — Market Regime Detection

Fetches the last 30 trading days of NIFTY 50 (`^NSEI`) and India VIX (`^INDIAVIX`) data via yfinance. Classifies market regime:

| Regime | Condition |
|---|---|
| `BULL` | 20d return > +2% and VIX < 20 |
| `BEAR` | 20d return < -2% or VIX > 25 |
| `SIDEWAYS` | otherwise |

### `news_agent.py` — FinBERT Sentiment

For each of the top-K tickers, fetches up to 10 recent headlines via the `yfinance` news API and scores each headline with `ProsusAI/finbert` (positive / neutral / negative). The per-stock sentiment score is the weighted average of headline scores (weight = FinBERT confidence). Results are cached in `data/news_cache.db` (SQLite, 1-hour TTL) to avoid redundant downloads.

> **First run:** FinBERT (~420 MB) is downloaded automatically to the HuggingFace cache (`~/.cache/huggingface/`). All subsequent runs load from disk.

### `risk_agent.py` — Risk Profiling

Computes per-stock risk metrics from the last 252 trading days of price data:

| Metric | Description |
|---|---|
| `vol_20d` | 20-day rolling return standard deviation (annualised) |
| `atr` | Average True Range (14-day) |
| `delta_52w_high` | % drawdown from 52-week high |
| `delta_52w_low` | % above 52-week low |
| `range_position` | Price position within 52-week range (0 = at low, 1 = at high) |
| `risk_tier` | `LOW` / `MEDIUM` / `HIGH` composite classification |

### `portfolio_agent.py` — Signal Fusion & Ranking

Fuses GNN predicted returns with news sentiment scores:

```
final_score = α × norm(gnn_score) + (1 − α) × norm(sentiment_score)
```

Ranks all stocks cross-sectionally by `final_score`, then assigns:
- `BUY` — top-K stocks
- `SELL` — bottom-K stocks
- `HOLD` — all others

### `report_agent.py` — Executive Research Note

Synthesises all upstream signals (regime, portfolio, risk, sentiment) into a structured Markdown research note. With `GOOGLE_API_KEY` set, uses Gemini Flash for the narrative. Without a key, generates a structured template summary.

---

## Project Structure

```
MultiAgent/
├── graph.py                    # LangGraph StateGraph — node wiring + edge conditions
├── orchestrator.py             # Pipeline coordinator (state initialisation, error handling)
├── run_pipeline.py             # CLI entry point
├── agents/
│   ├── gnn_agent.py            # GNN inference node
│   ├── macro_agent.py          # Macro regime detection node
│   ├── news_agent.py           # FinBERT sentiment node
│   ├── risk_agent.py           # Per-stock risk computation node
│   ├── portfolio_agent.py      # Signal fusion + BUY/SELL ranking node
│   └── report_agent.py         # Executive summary generation node
├── utils/
│   ├── finbert_loader.py       # HuggingFace FinBERT model + tokenizer loader
│   └── news_cache.py           # SQLite-backed news cache with TTL management
├── data/
│   └── news_cache.db           # Auto-created on first run
└── requirements.txt
```
