"""
run_pipeline.py — CLI entry point for the multi-agent forecasting pipeline.

Runs the LangGraph StateGraph by default. Use --no-graph to fall back to the
plain orchestrator (useful if langgraph is not installed).

Usage
-----
    # Full pipeline for today's date, top 10 stocks
    python run_pipeline.py

    # Specific date, GNN-only (no news, fast)
    python run_pipeline.py --date 2026-05-09 --no-news

    # Full pipeline with tuned alpha
    python run_pipeline.py --date 2026-05-09 --top-k 5 --alpha 0.6

    # Save output to CSV
    python run_pipeline.py --date 2026-05-09 --output results.csv

    # Print the Mermaid graph diagram and exit
    python run_pipeline.py --print-graph
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orchestrator import display


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-agent GNN + News portfolio pipeline.")
    p.add_argument("--date",        type=str,   default=str(date.today()),
                   help="Trading date (YYYY-MM-DD). Defaults to today.")
    p.add_argument("--top-k",       type=int,   default=10,
                   help="Number of long/short candidates to recommend.")
    p.add_argument("--alpha",       type=float, default=0.7,
                   help="GNN signal weight in [0, 1]. 0=news-only, 1=GNN-only.")
    p.add_argument("--no-news",     action="store_true",
                   help="Skip news sentiment agent (GNN predictions only, much faster).")
    p.add_argument("--checkpoint",  type=str,   default=None,
                   help="Path to hybrid model checkpoint (.dat). Auto-detected if not given.")
    p.add_argument("--device",      type=str,   default=None,
                   help="'cuda' or 'cpu'. Auto-detected if not given.")
    p.add_argument("--output",      type=str,   default=None,
                   help="Optional path to save the portfolio table as CSV.")
    p.add_argument("--max-headlines", type=int, default=20,
                   help="Max news headlines per ticker for FinBERT scoring.")
    p.add_argument("--no-graph",    action="store_true",
                   help="Use the plain orchestrator instead of LangGraph.")
    p.add_argument("--print-graph", action="store_true",
                   help="Print the Mermaid diagram for the LangGraph pipeline and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.print_graph:
        from graph import get_mermaid
        print(get_mermaid())
        return

    if args.no_graph:
        from orchestrator import run
        portfolio = run(
            date=args.date,
            top_k=args.top_k,
            alpha=args.alpha,
            max_news_headlines=args.max_headlines,
            checkpoint_path=args.checkpoint,
            device=args.device,
            no_news=args.no_news,
        )
    else:
        try:
            from graph import run_graph
        except ModuleNotFoundError as e:
            if e.name != "langgraph":
                raise
            print("LangGraph is not installed; falling back to the plain orchestrator.")
            from orchestrator import run
            portfolio = run(
                date=args.date,
                top_k=args.top_k,
                alpha=args.alpha,
                max_news_headlines=args.max_headlines,
                checkpoint_path=args.checkpoint,
                device=args.device,
                no_news=args.no_news,
            )
        else:
            portfolio = run_graph(
                date=args.date,
                top_k=args.top_k,
                alpha=args.alpha,
                no_news=args.no_news,
                max_news_headlines=args.max_headlines,
                checkpoint_path=args.checkpoint,
                device=args.device,
            )

    display(portfolio, top_k=args.top_k)

    if args.output:
        out = Path(args.output)
        portfolio.to_csv(out, index=True)
        print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
