"""
graph.py — LangGraph StateGraph for the multi-agent forecasting pipeline.

Nodes
-----
  gnn_node       : Run hybrid THGNN×MaGNet inference → predicted return per stock
  news_node      : Fetch yfinance headlines + FinBERT sentiment per stock
  portfolio_node : Fuse GNN rank + sentiment into a ranked buy/sell table

Routing
-------
  gnn_node → news_node → portfolio_node   (default)
  gnn_node → portfolio_node               (when no_news=True)

Usage
-----
    from graph import build_graph, run_graph

    portfolio_df = run_graph(date="2026-05-09", top_k=10, alpha=0.7)
"""
from __future__ import annotations

import operator
import time
from typing import Annotated

import pandas as pd
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from agents.gnn_agent import GNNAgent
from agents import news_agent as NewsAgent
from agents import portfolio_agent as PortfolioAgent


# ── State ────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # ---- inputs (set once at invocation) ----
    date:               str
    top_k:              int
    alpha:              float
    no_news:            bool
    checkpoint_path:    str | None
    device:             str | None
    max_news_headlines: int

    # ---- filled by nodes ----
    gnn_signals:    dict[str, float]        # ticker → predicted return
    news_signals:   dict[str, dict]         # ticker → {sentiment_score, headlines, ...}
    portfolio_rows: list[dict]              # DataFrame serialised as records

    # ---- append-only run log (each node appends its own lines) ----
    log: Annotated[list[str], operator.add]


# ── GNN agent singleton (avoids re-loading weights on every graph.invoke()) ──

_gnn_agent: GNNAgent | None = None

def _get_gnn_agent(checkpoint_path: str | None, device: str | None) -> GNNAgent:
    global _gnn_agent
    if _gnn_agent is None:
        _gnn_agent = GNNAgent(checkpoint_path=checkpoint_path, device=device)
    return _gnn_agent


# ── Nodes ─────────────────────────────────────────────────────────────────────

def gnn_node(state: PipelineState) -> dict:
    t0 = time.time()
    agent = _get_gnn_agent(state.get("checkpoint_path"), state.get("device"))
    signals = agent.predict(state["date"])
    elapsed = time.time() - t0
    return {
        "gnn_signals": signals,
        "log": [f"[gnn_node] {len(signals)} stocks predicted in {elapsed:.1f}s"],
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


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_after_gnn(state: PipelineState) -> str:
    """Skip the news node entirely when no_news=True."""
    return "portfolio_node" if state.get("no_news") else "news_node"


# ── Graph factory ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Compile and return the LangGraph StateGraph."""
    builder = StateGraph(PipelineState)

    builder.add_node("gnn_node",       gnn_node)
    builder.add_node("news_node",       news_node)
    builder.add_node("portfolio_node",  portfolio_node)

    builder.add_edge(START, "gnn_node")
    builder.add_conditional_edges(
        "gnn_node",
        _route_after_gnn,
        {"news_node": "news_node", "portfolio_node": "portfolio_node"},
    )
    builder.add_edge("news_node",      "portfolio_node")
    builder.add_edge("portfolio_node", END)

    return builder.compile()


# Module-level compiled graph (built once on import)
graph = build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_graph(
    date: str,
    top_k: int = 10,
    alpha: float = 0.7,
    no_news: bool = False,
    max_news_headlines: int = 20,
    checkpoint_path: str | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    """
    Invoke the LangGraph pipeline and return a portfolio DataFrame.
    Drop-in replacement for orchestrator.run().
    """
    initial_state: PipelineState = {
        "date":               date,
        "top_k":              top_k,
        "alpha":              alpha,
        "no_news":            no_news,
        "checkpoint_path":    checkpoint_path,
        "device":             device,
        "max_news_headlines": max_news_headlines,
        "gnn_signals":        {},
        "news_signals":       {},
        "portfolio_rows":     [],
        "log":                [],
    }

    print(f"\n{'='*60}")
    print(f"LangGraph Pipeline  |  date={date}  top_k={top_k}  alpha={alpha:.2f}")
    print(f"{'='*60}")

    final_state = graph.invoke(initial_state)

    for line in final_state.get("log", []):
        print(f"  {line}")

    rows = final_state.get("portfolio_rows", [])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def get_mermaid() -> str:
    """Return the Mermaid diagram string for this graph."""
    return graph.get_graph().draw_mermaid()
