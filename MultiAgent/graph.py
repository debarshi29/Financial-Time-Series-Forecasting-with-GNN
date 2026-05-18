"""
graph.py — LangGraph StateGraph for the multi-agent forecasting pipeline.

Nodes
-----
  gnn_node       : Run selected model variant inference → predicted return per stock
  news_node      : Fetch yfinance headlines + FinBERT sentiment per stock
  portfolio_node : Fuse GNN rank + sentiment into a ranked buy/sell table
  risk_node      : Compute beta/volatility/52w metrics for BUY+SELL candidates
  macro_node     : Fetch NIFTY 50 trend + India VIX; derive market regime + multiplier
  report_node    : Generate AI research note via Groq or Gemini (free LLMs)

Routing
-------
  gnn_node → news_node → portfolio_node → risk_node → macro_node → report_node
  gnn_node → portfolio_node  (when no_news=True)

Usage
-----
    from graph import build_graph, run_graph

    portfolio_df = run_graph(
        date="2026-05-09",
        top_k=10,
        alpha=0.7,
        model_variant="hybrid",   # "hybrid" | "mamba" | "thgnn"
        llm_provider="groq",      # "groq" | "gemini"  (or None to auto-detect)
    )
"""
from __future__ import annotations

import copy
import operator
import os
import time
from pathlib import Path
from typing import Annotated

# Load .env from project root or demo/ directory (whichever exists first)
try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parents[1]
    load_dotenv(_root / ".env", override=False)
    load_dotenv(_root / "demo" / ".env", override=False)
except ImportError:
    pass

import pandas as pd
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from agents.gnn_agent import GNNAgent
from agents import news_agent as NewsAgent
from agents import portfolio_agent as PortfolioAgent
from agents.risk_agent import RiskAgent
from agents.macro_agent import MarketContextAgent


# ── State ────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # ---- inputs (set once at invocation) ----
    date:               str
    top_k:              int
    alpha:              float
    no_news:            bool
    no_report:          bool            # skip LLM report generation
    model_variant:      str             # "hybrid" | "mamba" | "thgnn"
    llm_provider:       str | None      # "groq" | "gemini" | None (auto-detect)
    checkpoint_path:    str | None
    device:             str | None
    max_news_headlines: int

    # ---- filled by nodes ----
    gnn_signals:    dict[str, float]    # ticker → predicted return score
    news_signals:   dict[str, dict]     # ticker → {sentiment_score, headlines, ...}
    portfolio_rows: list[dict]          # DataFrame serialised as records
    risk_data:      dict[str, dict]     # ticker → {beta, vol_20d, risk_label, ...}
    macro_context:  dict               # {market_regime, vix_level, confidence_multiplier, ...}
    report_markdown: str               # final AI-generated markdown report
    report_path:    str               # path to saved .md file

    # ---- append-only run log ----
    log: Annotated[list[str], operator.add]


# ── Agent singletons ──────────────────────────────────────────────────────────

_gnn_agent: GNNAgent | None = None
_risk_agent: RiskAgent | None = None
_macro_agent: MarketContextAgent | None = None


def _get_gnn_agent(checkpoint_path: str | None, device: str | None, variant: str) -> GNNAgent:
    global _gnn_agent
    if _gnn_agent is None or _gnn_agent.variant != variant:
        _gnn_agent = GNNAgent(
            checkpoint_path=checkpoint_path,
            device=device,
            model_variant=variant,  # type: ignore[arg-type]
        )
    return _gnn_agent


def _get_risk_agent() -> RiskAgent:
    global _risk_agent
    if _risk_agent is None:
        _risk_agent = RiskAgent()
    return _risk_agent


def _get_macro_agent() -> MarketContextAgent:
    global _macro_agent
    if _macro_agent is None:
        _macro_agent = MarketContextAgent()
    return _macro_agent


# ── Nodes ─────────────────────────────────────────────────────────────────────

def gnn_node(state: PipelineState) -> dict:
    t0 = time.time()
    variant = state.get("model_variant", "hybrid")
    agent = _get_gnn_agent(state.get("checkpoint_path"), state.get("device"), variant)
    signals = agent.predict(state["date"])
    elapsed = time.time() - t0
    return {
        "gnn_signals": signals,
        "log": [f"[gnn_node] variant={variant}  {len(signals)} stocks in {elapsed:.1f}s"],
    }


def news_node(state: PipelineState) -> dict:
    t0 = time.time()
    tickers = list(state["gnn_signals"].keys())
    signals = NewsAgent.score_tickers(
        tickers,
        max_headlines=state.get("max_news_headlines", 20),
        device=state.get("device"),
    )
    covered = sum(1 for v in signals.values() if v["news_count"] > 0)
    elapsed = time.time() - t0
    return {
        "news_signals": signals,
        "log": [f"[news_node] news found for {covered}/{len(tickers)} tickers in {elapsed:.1f}s"],
    }


def portfolio_node(state: PipelineState) -> dict:
    t0 = time.time()
    df = PortfolioAgent.construct(
        gnn_signals=state["gnn_signals"],
        news_signals=state.get("news_signals", {}),
        alpha=state.get("alpha", 0.7),
        top_k=state.get("top_k", 10),
    )
    elapsed = time.time() - t0
    buys  = (df["action"] == "BUY").sum()
    sells = (df["action"] == "SELL").sum()
    return {
        "portfolio_rows": df.to_dict(orient="records"),
        "log": [f"[portfolio_node] {buys} BUY / {sells} SELL in {elapsed:.1f}s"],
    }


def risk_node(state: PipelineState) -> dict:
    t0 = time.time()
    rows = state.get("portfolio_rows", [])
    if not rows:
        return {"risk_data": {}, "log": ["[risk_node] no portfolio rows, skipping"]}

    df = pd.DataFrame(rows)
    candidates = df[df["action"].isin(["BUY", "SELL"])]["ticker"].tolist()
    if not candidates:
        return {"risk_data": {}, "log": ["[risk_node] no BUY/SELL candidates, skipping"]}

    agent = _get_risk_agent()
    risk_data = agent.assess(candidates, end_date=state.get("date"))
    elapsed = time.time() - t0
    return {
        "risk_data": risk_data,
        "log": [f"[risk_node] assessed {len(risk_data)} tickers in {elapsed:.1f}s"],
    }


def macro_node(state: PipelineState) -> dict:
    t0 = time.time()
    agent = _get_macro_agent()
    macro = agent.fetch(state["date"])
    elapsed = time.time() - t0

    # Apply confidence multiplier to BUY scores
    multiplier = macro.get("confidence_multiplier", 1.0)
    rows = state.get("portfolio_rows", [])
    if rows and multiplier != 1.0:
        rows = copy.deepcopy(rows)
        for row in rows:
            base = row.get("final_score", 0.0)
            row["adjusted_score"] = round(base * multiplier if row.get("action") == "BUY" else base, 4)
    elif rows:
        rows = copy.deepcopy(rows)
        for row in rows:
            row["adjusted_score"] = row.get("final_score", 0.0)

    return {
        "macro_context":  macro,
        "portfolio_rows": rows,
        "log": [
            f"[macro_node] regime={macro['market_regime']}  "
            f"VIX={macro['vix_level']:.1f}  multiplier={multiplier:.2f} in {elapsed:.1f}s"
        ],
    }


def report_node(state: PipelineState) -> dict:
    t0 = time.time()

    if state.get("no_report"):
        return {"report_markdown": "", "report_path": "", "log": ["[report_node] skipped (no_report=True)"]}

    provider = state.get("llm_provider") or _auto_detect_provider()
    if provider is None:
        return {
            "report_markdown": "",
            "report_path": "",
            "log": ["[report_node] skipped (no GROQ_API_KEY or GOOGLE_API_KEY found)"],
        }

    try:
        from agents.report_agent import ReportWriterAgent
        agent = ReportWriterAgent(provider=provider)
        portfolio_df = pd.DataFrame(state.get("portfolio_rows", []))
        markdown = agent.generate(
            date_str=state["date"],
            portfolio_df=portfolio_df,
            news_signals=state.get("news_signals", {}),
            risk_data=state.get("risk_data", {}),
            macro_context=state.get("macro_context", {}),
            model_variant=state.get("model_variant", "hybrid"),
            top_k=state.get("top_k", 5),
            save=True,
        )
        elapsed = time.time() - t0
        variant = state.get("model_variant", "hybrid")
        report_path = str(
            (lambda p: p / "reports" / f"{state['date']}_{variant}_report.md")(
                __import__("pathlib").Path(__file__).resolve().parents[1]
            )
        )
        return {
            "report_markdown": markdown,
            "report_path":     report_path,
            "log": [f"[report_node] {len(markdown)} chars generated in {elapsed:.1f}s"],
        }
    except Exception as e:
        return {
            "report_markdown": "",
            "report_path":     "",
            "log": [f"[report_node] FAILED: {e}"],
        }


def _auto_detect_provider() -> str | None:
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return None


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_after_gnn(state: PipelineState) -> str:
    return "portfolio_node" if state.get("no_news") else "news_node"


# ── Graph factory ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(PipelineState)

    builder.add_node("gnn_node",       gnn_node)
    builder.add_node("news_node",      news_node)
    builder.add_node("portfolio_node", portfolio_node)
    builder.add_node("risk_node",      risk_node)
    builder.add_node("macro_node",     macro_node)
    builder.add_node("report_node",    report_node)

    builder.add_edge(START, "gnn_node")
    builder.add_conditional_edges(
        "gnn_node",
        _route_after_gnn,
        {"news_node": "news_node", "portfolio_node": "portfolio_node"},
    )
    builder.add_edge("news_node",      "portfolio_node")
    builder.add_edge("portfolio_node", "risk_node")
    builder.add_edge("risk_node",      "macro_node")
    builder.add_edge("macro_node",     "report_node")
    builder.add_edge("report_node",    END)

    return builder.compile()


graph = build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_graph(
    date: str,
    top_k: int = 10,
    alpha: float = 0.7,
    no_news: bool = False,
    no_report: bool = False,
    model_variant: str = "hybrid",
    llm_provider: str | None = None,
    max_news_headlines: int = 20,
    checkpoint_path: str | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    """
    Run the full multi-agent pipeline and return a portfolio DataFrame.

    Parameters
    ----------
    date           : Trading date 'YYYY-MM-DD'
    top_k          : Number of BUY + SELL candidates to surface
    alpha          : GNN signal weight (0=news-only, 1=GNN-only)
    no_news        : Skip FinBERT news sentiment (faster)
    no_report      : Skip LLM report generation
    model_variant  : 'hybrid' | 'mamba' | 'thgnn'
    llm_provider   : 'groq' | 'gemini' | None (auto-detect from env vars)
    """
    initial_state: PipelineState = {
        "date":               date,
        "top_k":              top_k,
        "alpha":              alpha,
        "no_news":            no_news,
        "no_report":          no_report,
        "model_variant":      model_variant,
        "llm_provider":       llm_provider,
        "checkpoint_path":    checkpoint_path,
        "device":             device,
        "max_news_headlines": max_news_headlines,
        "gnn_signals":        {},
        "news_signals":       {},
        "portfolio_rows":     [],
        "risk_data":          {},
        "macro_context":      {},
        "report_markdown":    "",
        "report_path":        "",
        "log":                [],
    }

    print(f"\n{'='*60}")
    print(f"LangGraph Pipeline  |  date={date}  model={model_variant}  top_k={top_k}")
    print(f"{'='*60}")

    final_state = graph.invoke(initial_state)

    for line in final_state.get("log", []):
        print(f"  {line}")

    rows = final_state.get("portfolio_rows", [])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def run_graph_full(
    date: str,
    **kwargs,
) -> dict:
    """Like run_graph() but returns the full final state dict (for Streamlit)."""
    initial_state: PipelineState = {
        "date":               date,
        "top_k":              kwargs.get("top_k", 10),
        "alpha":              kwargs.get("alpha", 0.7),
        "no_news":            kwargs.get("no_news", False),
        "no_report":          kwargs.get("no_report", False),
        "model_variant":      kwargs.get("model_variant", "hybrid"),
        "llm_provider":       kwargs.get("llm_provider", None),
        "checkpoint_path":    kwargs.get("checkpoint_path", None),
        "device":             kwargs.get("device", None),
        "max_news_headlines": kwargs.get("max_news_headlines", 20),
        "gnn_signals":        {},
        "news_signals":       {},
        "portfolio_rows":     [],
        "risk_data":          {},
        "macro_context":      {},
        "report_markdown":    "",
        "report_path":        "",
        "log":                [],
    }
    return graph.invoke(initial_state)


def get_mermaid() -> str:
    return graph.get_graph().draw_mermaid()
