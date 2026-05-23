"""
FastAPI backend for the GNN+News demo.

Run (from project root):
    uvicorn demo.api:app --reload --port 8000

Or:
    python -m uvicorn demo.api:app --reload --port 8000
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MultiAgent"))
sys.path.insert(0, str(ROOT / "THGNN_MaGNet"))

app = FastAPI(title="GNN+News Demo API")

# Serve built React frontend if available
_static = Path(__file__).parent / "static"
if _static.exists():
    app.mount("/assets", StaticFiles(directory=str(_static / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{_:path}", include_in_schema=False)
    async def spa(_: str = ""):
        return FileResponse(str(_static / "index.html"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_backtest() -> list[dict]:
    results_dir = ROOT / "THGNN" / "data" / "backtest_results"
    out: list[dict] = []
    if not results_dir.exists():
        return out
    for folder in sorted(results_dir.iterdir()):
        mj = folder / "metrics.json"
        mt = folder / "metrics_report.txt"
        if mj.exists():
            import json
            data = json.loads(mj.read_text())
            data["source"] = "hybrid" if "hybrid" in folder.name.lower() else "thgnn"
            data["folder"] = folder.name
            out.append(data)
        elif mt.exists():
            text = mt.read_text()
            entry: dict = {"folder": folder.name, "source": "thgnn", "longshort": {}}
            for line in text.splitlines():
                if "Total Return" in line and "%" in line:
                    try: entry["longshort"]["total_return_pct"] = float(line.split()[-1])
                    except Exception: pass
                if "Sharpe Ratio" in line:
                    try: entry["longshort"]["sharpe"] = float(line.split()[-1])
                    except Exception: pass
            out.append(entry)
    return out


# ── schemas ───────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    date:          str
    top_k:         int   = 10
    alpha:         float = 0.7
    no_news:       bool  = False
    no_report:     bool  = False
    model_variant: str   = "hybrid"
    llm_provider:  str | None = None


class NewsRequest(BaseModel):
    ticker:        str
    max_headlines: int = 15


# ── routes ────────────────────────────────────────────────────────────────────

@app.post("/api/run")
async def run_pipeline(req: RunRequest) -> dict[str, Any]:
    try:
        from graph import run_graph_full  # type: ignore
        result = run_graph_full(
            date=req.date,
            top_k=req.top_k,
            alpha=req.alpha,
            no_news=req.no_news,
            no_report=req.no_report,
            model_variant=req.model_variant,
            llm_provider=req.llm_provider,
        )
        return {
            "success":         True,
            "portfolio":       result.get("portfolio_rows", []),
            "risk_data":       result.get("risk_data", {}),
            "macro_context":   result.get("macro_context", {}),
            "report_markdown": result.get("report_markdown", ""),
            "news_signals":    result.get("news_signals", {}),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@app.get("/api/backtest")
async def get_backtest() -> list[dict]:
    return _load_backtest()


@app.get("/api/tickers")
async def get_tickers() -> list[str]:
    f = ROOT / "THGNN" / "data" / "valid_nifty500.txt"
    return f.read_text().strip().splitlines() if f.exists() else []


@app.post("/api/news")
async def get_news(req: NewsRequest) -> dict[str, Any]:
    try:
        from utils.finbert_loader import get_finbert  # type: ignore
        get_finbert()
        from agents.news_agent import score_ticker  # type: ignore
        result = score_ticker(req.ticker, max_headlines=req.max_headlines, cache_max_age_s=600)
        return {"success": True, **result}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@app.get("/api/reports/latest")
async def get_latest_report() -> dict[str, Any]:
    saved = sorted(glob.glob(str(ROOT / "reports" / "*_report.md")), reverse=True)
    if saved:
        content = Path(saved[0]).read_text(encoding="utf-8")
        return {"content": content, "filename": Path(saved[0]).name}
    return {"content": None, "filename": None}
