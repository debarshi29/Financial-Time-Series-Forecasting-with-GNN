"""
Streamlit demo — Multi-Agent GNN + News Financial Forecasting System

Tabs:
  1. Live Portfolio  — Run the pipeline for a chosen date
  2. Model Comparison — Backtest metrics: THGNN vs Hybrid vs Hybrid+News
  3. News Drill-Down  — Per-stock news headlines + FinBERT sentiment
  4. System Overview  — Architecture explanation

Run:
    cd demo
    streamlit run app.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Resolve project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MultiAgent"))
sys.path.insert(0, str(ROOT / "THGNN_MaGNet"))

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GNN+News Stock Forecasting",
    page_icon="📈",
    layout="wide",
)

# ── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading GNN model (one-time)...")
def _load_gnn_agent(checkpoint: str | None = None):
    from agents.gnn_agent import GNNAgent
    return GNNAgent(checkpoint_path=checkpoint)


@st.cache_resource(show_spinner="Loading FinBERT model (one-time, ~420MB)...")
def _load_finbert():
    from utils.finbert_loader import get_finbert
    get_finbert()


def _run_pipeline(run_date: str, top_k: int, alpha: float, no_news: bool):
    from graph import run_graph
    return run_graph(
        date=run_date,
        top_k=top_k,
        alpha=alpha,
        no_news=no_news,
    )


def _color_action(val: str) -> str:
    if val == "BUY":
        return "background-color: #d4edda; color: #155724; font-weight: bold"
    if val == "SELL":
        return "background-color: #f8d7da; color: #721c24; font-weight: bold"
    return ""


def _load_backtest_results() -> list[dict]:
    results_dir = ROOT / "THGNN" / "data" / "backtest_results"
    out = []
    if not results_dir.exists():
        return out
    for folder in sorted(results_dir.iterdir()):
        metrics_json = folder / "metrics.json"
        metrics_txt  = folder / "metrics_report.txt"
        if metrics_json.exists():
            data = json.loads(metrics_json.read_text())
            data["source"] = "hybrid" if "hybrid" in folder.name.lower() else "thgnn"
            data["folder"] = folder.name
            out.append(data)
        elif metrics_txt.exists():
            # Legacy text format — parse key lines
            text = metrics_txt.read_text()
            entry = {"folder": folder.name, "source": "thgnn", "longshort": {}}
            for line in text.splitlines():
                if "Total Return" in line and "%" in line:
                    try:
                        entry["longshort"]["total_return_pct"] = float(line.split()[-1])
                    except Exception:
                        pass
                if "Sharpe Ratio" in line:
                    try:
                        entry["longshort"]["sharpe"] = float(line.split()[-1])
                    except Exception:
                        pass
            out.append(entry)
    return out


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    chosen_date = st.date_input("Date", value=date.today() - timedelta(days=1),
                                min_value=date(2024, 1, 1), max_value=date.today())
    top_k = st.slider("Top-K recommendations", 3, 20, 10)
    alpha = st.slider("GNN weight (alpha)", 0.0, 1.0, 0.7, 0.05,
                      help="1.0 = pure GNN, 0.0 = pure news sentiment")
    no_news = st.checkbox("Skip news (GNN-only, faster)", value=False)
    st.divider()
    st.caption("Checkpoint auto-detected from THGNN_MaGNet/data/model_saved/")


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Portfolio",
    "📈 Model Comparison",
    "📰 News Drill-Down",
    "🏗 System Overview",
])


# ── Tab 1: Live Portfolio ────────────────────────────────────────────────────
with tab1:
    st.header(f"Portfolio Recommendations — {chosen_date}")
    st.caption(
        f"alpha={alpha:.2f}  |  top_k={top_k}  |  "
        f"{'GNN-only' if no_news else 'GNN + News (FinBERT)'}"
    )

    run_col, _ = st.columns([1, 4])
    if run_col.button("Run Pipeline", type="primary"):
        with st.spinner("Running multi-agent pipeline..."):
            try:
                df = _run_pipeline(str(chosen_date), top_k, alpha, no_news)
                st.session_state["portfolio"] = df
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.exception(e)

    if "portfolio" in st.session_state:
        df = st.session_state["portfolio"]
        buys  = df[df["action"] == "BUY"]
        sells = df[df["action"] == "SELL"]

        c1, c2, c3 = st.columns(3)
        c1.metric("BUY candidates",  len(buys))
        c2.metric("SELL candidates", len(sells))
        c3.metric("Total stocks",    len(df))

        st.subheader("Full Ranking")
        styled = df.style.applymap(_color_action, subset=["action"])
        st.dataframe(styled, use_container_width=True, height=500)

        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "portfolio.csv", "text/csv")

        if "sentiment_score" in df.columns and not no_news:
            st.subheader("GNN Score vs News Sentiment")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = {"BUY": "#2ecc71", "SELL": "#e74c3c", "HOLD": "#95a5a6"}
            for action, grp in df.groupby("action"):
                ax.scatter(grp["gnn_rank"], grp["sentiment_score"],
                           label=action, c=colors.get(action, "gray"), s=60, alpha=0.8)
            ax.set_xlabel("GNN Rank Score (normalised)")
            ax.set_ylabel("News Sentiment Score")
            ax.set_title("GNN vs Sentiment Signal Space")
            ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)
    else:
        st.info("Click **Run Pipeline** to generate recommendations.")


# ── Tab 2: Model Comparison ──────────────────────────────────────────────────
with tab2:
    st.header("Backtest Performance Comparison")

    results = _load_backtest_results()

    if not results:
        st.warning(
            "No backtest results found in `THGNN/data/backtest_results/`. "
            "Run `python THGNN/backtest.py` to generate them."
        )
        st.code("cd THGNN\npython backtest.py --start-date 2025-01-01 --end-date 2026-02-28 --top-k 5")
    else:
        rows = []
        for r in results:
            ls = r.get("longshort", {})
            ic = r.get("ic", {})
            rows.append({
                "Run": r.get("folder", "?"),
                "Model": r.get("source", "?").upper(),
                "Total Return (%)": ls.get("total_return_pct"),
                "Ann. Return (%)":  ls.get("ann_return_pct"),
                "Sharpe":           ls.get("sharpe"),
                "Max DD (%)":       ls.get("max_drawdown_pct"),
                "Win Rate (%)":     ls.get("win_rate_pct"),
                "IC Mean":          ic.get("ic_mean") if ic else None,
                "IC t-stat":        ic.get("ic_tstat") if ic else None,
            })
        compare_df = pd.DataFrame(rows)
        st.dataframe(compare_df, use_container_width=True)

        # Bar chart: Sharpe ratios
        valid_sharpe = compare_df.dropna(subset=["Sharpe"])
        if not valid_sharpe.empty:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            for ax, metric in zip(axes, ["Sharpe", "Total Return (%)", "IC Mean"]):
                sub = compare_df.dropna(subset=[metric])
                if sub.empty:
                    ax.set_visible(False)
                    continue
                colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sub[metric]]
                ax.bar(sub["Run"], sub[metric], color=colors, alpha=0.85)
                ax.set_title(metric); ax.set_xticklabels(sub["Run"], rotation=20, ha="right")
                ax.axhline(0, color="gray", lw=0.8, ls=":")
                ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    # Show equity curve images if available
    st.subheader("Equity Curves")
    equity_images = list((ROOT / "THGNN" / "data" / "backtest_results").glob("*/equity_curve.png"))
    if equity_images:
        for img_path in sorted(equity_images)[:3]:
            st.image(str(img_path), caption=img_path.parent.name, use_column_width=True)
    else:
        st.info("Equity curve images will appear here after running `python THGNN/backtest.py`.")


# ── Tab 3: News Drill-Down ───────────────────────────────────────────────────
with tab3:
    st.header("News Sentiment Drill-Down")

    tickers_file = ROOT / "THGNN" / "data" / "valid_nifty500.txt"
    if tickers_file.exists():
        all_tickers = tickers_file.read_text().strip().splitlines()
    else:
        all_tickers = []

    selected_ticker = st.selectbox("Select a stock", all_tickers or ["No tickers found"])
    max_h = st.slider("Max headlines to fetch", 5, 30, 15)

    if st.button("Fetch & Score News", type="primary"):
        with st.spinner(f"Fetching news for {selected_ticker}..."):
            try:
                _load_finbert()
                from agents.news_agent import score_ticker
                result = score_ticker(selected_ticker, max_headlines=max_h, cache_max_age_s=600)
                st.session_state["news_result"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

    if "news_result" in st.session_state:
        r = st.session_state["news_result"]
        st.metric("Aggregate Sentiment Score", f"{r['sentiment_score']:+.3f}",
                  delta=f"{r['news_count']} headlines")

        sentiment_label = (
            "POSITIVE" if r["sentiment_score"] > 0.1 else
            "NEGATIVE" if r["sentiment_score"] < -0.1 else "NEUTRAL"
        )
        color_map = {"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "gray"}
        st.markdown(
            f"**Overall Sentiment: :{color_map[sentiment_label]}[{sentiment_label}]**"
        )

        if r["headlines"]:
            st.subheader("Headlines & FinBERT Scores")
            headlines_df = pd.DataFrame(r["headlines"])
            if not headlines_df.empty:
                # Style bar-like background
                def _sentiment_color(row):
                    if row["positive"] > row["negative"] and row["positive"] > row["neutral"]:
                        return ["background-color: #d4edda"] * len(row)
                    if row["negative"] > row["positive"] and row["negative"] > row["neutral"]:
                        return ["background-color: #f8d7da"] * len(row)
                    return [""] * len(row)
                st.dataframe(
                    headlines_df.style.apply(_sentiment_color, axis=1),
                    use_container_width=True
                )
        else:
            st.info(f"No news found for {selected_ticker} via yfinance.")


# ── Tab 4: System Overview ───────────────────────────────────────────────────
with tab4:
    st.header("Multi-Agent System Architecture")

    # Live LangGraph diagram
    st.subheader("LangGraph Pipeline")
    try:
        from graph import get_mermaid
        mermaid_src = get_mermaid()
        st.markdown(f"```mermaid\n{mermaid_src}\n```")
        st.caption("Auto-generated from the compiled LangGraph StateGraph. "
                   "The dashed branch (gnn_node → portfolio_node) is taken when --no-news is set.")
    except Exception as e:
        st.warning(f"Could not render LangGraph diagram: {e}")
        st.markdown("""
```
START → gnn_node → news_node → portfolio_node → END
                ↘ (no_news)  ↗
```
""")

    st.markdown("""
    ## State flowing through the graph

    Each node reads from and writes to a shared `PipelineState` TypedDict:

    | Field | Set by | Type |
    |-------|--------|------|
    | `date`, `top_k`, `alpha`, `no_news` | caller | inputs |
    | `gnn_signals` | `gnn_node` | `dict[ticker → score]` |
    | `news_signals` | `news_node` | `dict[ticker → {sentiment, headlines}]` |
    | `portfolio_rows` | `portfolio_node` | `list[dict]` (DataFrame records) |
    | `log` | all nodes | append-only list of timing strings |

    ## Model Architecture

    ### THGNN (Temporal Heterogeneous Graph Neural Network)
    - **GRU** temporal encoder per stock
    - **Positive/Negative GAT** layers on correlation graph
    - **Semantic attention fusion** + PairNorm

    ### Hybrid THGNN × MaGNet Extension
    - **MAGE**: BiGRU-lite + SparseMoE + Multi-Head Attention
    - **TCH**: Temporal Causal Hypergraph (lead-lag discovery)
    - **GPH**: Global Probabilistic Hypergraph (latent market themes)
    - **4-stream semantic fusion**: MAGE + TCH + PosGAT + NegGAT

    ### News Sentiment (FinBERT)
    - `ProsusAI/finbert` — BERT fine-tuned on financial text
    - Input: recent headlines from yfinance (free, no API key)
    - Output: positive / negative / neutral probability per headline
    - Aggregate: `sentiment_score = mean(positive) − mean(negative)`

    ### Signal Fusion
    ```
    final_score = α × GNN_rank + (1−α) × sentiment_norm
    ```
    where `GNN_rank` is the cross-sectional percentile rank and
    `sentiment_norm` is z-score normalised sentiment (clipped to [0,1]).
    """)

    # Show walk-forward results if available
    wf_plot = ROOT / "THGNN" / "data" / "plots" / "walk_forward_results.png"
    if wf_plot.exists():
        st.subheader("Walk-Forward Validation Results (THGNN)")
        st.image(str(wf_plot), use_column_width=True)
