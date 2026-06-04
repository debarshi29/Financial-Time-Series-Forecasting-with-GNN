# Demo — Trading Terminal Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Plotly.js](https://img.shields.io/badge/Plotly.js-2.x-3F4F75)](https://plotly.com/javascript/)

A self-contained, dark-themed trading-terminal web dashboard that provides an interactive front-end to the six-agent LangGraph pipeline. The entire UI is served by a lightweight Python HTTP server — no Node.js, no bundler, no external dependencies beyond the Python environment.

---

## Quick Start

```bash
cd demo
python server.py
# Navigate to http://localhost:8000
```

---

## Interface

The dashboard is a single-page application styled as a professional trading terminal. Five tabs are available:

### Portfolio

The primary tab. Select a date, configure the `α` fusion weight and top-K count, then click **Run Pipeline** to execute the full six-agent LangGraph pipeline. Results render as:

- An interactive **BUY/SELL signal table** with GNN score, sentiment, risk tier, and recommended action
- A **scatter plot** of GNN score vs. sentiment score, coloured by action

### News

Per-ticker FinBERT sentiment drill-down. Select a ticker from the dropdown and click **Fetch & Score** to retrieve the latest headlines from yfinance, score each with FinBERT, and display:

- Aggregate sentiment score and label (POSITIVE / NEUTRAL / NEGATIVE)
- Per-headline sentiment with article titles and source links

### Overview

Static reference tab showing:

- The multi-agent system architecture diagram
- Walk-forward validation summary and key model metrics

### Risk

Per-stock risk metrics table (populated after running the pipeline). Shows 20-day volatility, ATR, 52-week range position, delta from 52-week high/low, and composite risk tier.

### Report

The AI-generated daily research note produced by the report agent. Rendered as formatted Markdown with a one-click copy-to-clipboard action.

---

## Server Architecture

```
server.py  ──  Python stdlib http.server.BaseHTTPRequestHandler
    │
    ├── GET  /                    → serves templates/index.html
    ├── GET  /api/tickers         → returns valid_nifty500.txt ticker list
    ├── POST /api/run             → runs the LangGraph pipeline, streams JSON result
    └── POST /api/news            → fetches + scores news for a single ticker
```

No framework dependencies — the server uses only Python's standard `http.server` module. The frontend communicates exclusively via `fetch()` against these four endpoints.

---

## File Reference

| File | Purpose |
|---|---|
| `server.py` | Lightweight Python HTTP server; routes and API handlers |
| `api.py` | Pipeline invocation logic called by `server.py` API handlers |
| `templates/index.html` | Complete single-page trading terminal UI (HTML + CSS + JS) |
| `static/` | Static assets (fonts, icons) if present |
| `.env` | Local environment overrides (e.g. `PORT`, `GOOGLE_API_KEY`) |

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | HTTP port to bind |
| `GOOGLE_API_KEY` | — | Enables Gemini-powered research notes in the Report tab |

Set these in `demo/.env` or export them before running `python server.py`.
