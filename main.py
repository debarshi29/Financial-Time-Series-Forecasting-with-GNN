"""
Financial Time Series Forecasting with GNN
===========================================
Entry point — prints project info and available commands.

For actual usage, see:
  • THGNN/train_ic_ranked.py        — baseline model training
  • THGNN_MaGNet/train_hybrid.py    — hybrid model training
  • MultiAgent/run_pipeline.py      — live buy/sell signals
  • demo/app.py                     — Streamlit dashboard
"""

import sys

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   Financial Time Series Forecasting with GNN                ║
║   Graph-Structured Deep Learning · NIFTY 50/500             ║
╚══════════════════════════════════════════════════════════════╝

Available components
────────────────────
  THGNN/           Baseline model (GRU + Pos/Neg GAT)
  THGNN_MaGNet/    Hybrid model   (MaGNet + TCH + GPH)
  THGNN_Mamba_MoE/ Experimental   (Mamba SSM + MoE)
  MultiAgent/      LangGraph pipeline (GNN + FinBERT + Portfolio)
  demo/            Streamlit dashboard

Quick start
────────────
  # Backtest the hybrid model
  cd THGNN_MaGNet && python backtest_hybrid.py --start-date 2024-01-01 --end-date 2026-02-28

  # Run the multi-agent pipeline for today
  cd MultiAgent && python run_pipeline.py --date 2026-05-27 --top-k 10

  # Launch the interactive demo
  cd demo && streamlit run app.py

See README.md for full documentation.
"""


def main() -> int:
    print(BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
