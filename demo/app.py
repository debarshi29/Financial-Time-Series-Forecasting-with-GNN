"""
Streamlit demo — Multi-Agent GNN + News Financial Forecasting System

Tabs:
  1. Live Portfolio  — Run the pipeline for a chosen date
  2. Model Comparison — Backtest metrics: THGNN vs Hybrid vs Hybrid+News
  3. News Drill-Down  — Per-stock news headlines + FinBERT sentiment
  4. System Overview  — Architecture explanation
  5. Risk Dashboard   — Beta / volatility / 52-week risk metrics
  6. Daily Report     — AI-generated research note (Groq or Gemini)

Run:
    cd demo
    streamlit run app.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MultiAgent"))
sys.path.insert(0, str(ROOT / "THGNN_MaGNet"))

st.set_page_config(
    page_title="GNN+News Stock Forecasting",
    page_icon="📈",
    layout="wide",
)

# ── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading GNN model (one-time)...")
def _load_gnn_agent(variant: str, checkpoint: str | None = None):
    from agents.gnn_agent import GNNAgent
    return GNNAgent(checkpoint_path=checkpoint, model_variant=variant)


@st.cache_resource(show_spinner="Loading FinBERT model (one-time, ~420MB)...")
def _load_finbert():
    from utils.finbert_loader import get_finbert
    get_finbert()


def _run_pipeline(run_date: str, top_k: int, alpha: float, no_news: bool,
                  model_variant: str, no_report: bool, llm_provider: str | None):
    from graph import run_graph_full
    return run_graph_full(
        date=run_date,
        top_k=top_k,
        alpha=alpha,
        no_news=no_news,
        no_report=no_report,
        model_variant=model_variant,
        llm_provider=llm_provider,
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
    chosen_date = st.date_input(
        "Date", value=date.today() - timedelta(days=1),
        min_value=date(2015, 1, 1), max_value=date.today(),
    )
    top_k  = st.slider("Top-K recommendations", 3, 20, 10)
    alpha  = st.slider("GNN weight (alpha)", 0.0, 1.0, 0.7, 0.05,
                       help="1.0 = pure GNN signal, 0.0 = pure news sentiment")
    st.divider()

    model_variant = st.selectbox(
        "Model variant",
        ["hybrid", "mamba", "thgnn"],
        format_func=lambda v: {
            "hybrid": "Hybrid (BiGRU + MoE + Hypergraph)",
            "mamba":  "Mamba + MoE",
            "thgnn":  "Base THGNN (GRU + GAT)",
        }[v],
    )
    no_news   = st.checkbox("Skip news (GNN-only, faster)", value=False)
    no_report = st.checkbox("Skip AI report", value=False)
    llm_choice = st.radio("LLM provider", ["Auto-detect", "Groq", "Gemini"], index=0,
                          help="Groq: set GROQ_API_KEY  |  Gemini: set GOOGLE_API_KEY")
    llm_provider = None if llm_choice == "Auto-detect" else llm_choice.lower()
    st.divider()
    st.caption("Checkpoints auto-detected from model_saved/ dirs")


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Live Portfolio",
    "📈 Model Comparison",
    "📰 News Drill-Down",
    "🏗 System Overview",
    "⚠️ Risk Dashboard",
    "📝 Daily Report",
])


# ── Tab 1: Live Portfolio ────────────────────────────────────────────────────
with tab1:
    st.header(f"Portfolio Recommendations — {chosen_date}")
    st.caption(
        f"model={model_variant}  alpha={alpha:.2f}  top_k={top_k}  "
        f"{'GNN-only' if no_news else 'GNN + News (FinBERT)'}"
    )

    run_col, _ = st.columns([1, 4])
    if run_col.button("Run Pipeline", type="primary"):
        with st.spinner("Running multi-agent pipeline..."):
            try:
                final_state = _run_pipeline(
                    str(chosen_date), top_k, alpha, no_news,
                    model_variant, no_report, llm_provider,
                )
                st.session_state["final_state"]  = final_state
                st.session_state["portfolio"]    = pd.DataFrame(final_state.get("portfolio_rows", []))
                st.session_state["risk_data"]    = final_state.get("risk_data", {})
                st.session_state["macro_context"] = final_state.get("macro_context", {})
                st.session_state["report_markdown"] = final_state.get("report_markdown", "")
                st.session_state["news_signals"] = final_state.get("news_signals", {})
                st.session_state["run_date"]     = str(chosen_date)
                st.session_state["model_variant"] = model_variant
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.exception(e)

    # Macro context banner
    if "macro_context" in st.session_state and st.session_state["macro_context"]:
        mc = st.session_state["macro_context"]
        regime = mc.get("market_regime", "SIDEWAYS")
        vix    = mc.get("vix_level", 0.0)
        mult   = mc.get("confidence_multiplier", 1.0)
        nifty  = mc.get("nifty_return_1d", 0.0)
        trend5 = mc.get("nifty_trend_5d", 0.0)
        colors = {"BULL": "green", "BEAR": "red", "SIDEWAYS": "orange"}
        c = colors.get(regime, "gray")
        st.markdown(
            f"**Market Regime: :{c}[{regime}]** &nbsp;|&nbsp; "
            f"NIFTY 1d: **{nifty:+.2%}** &nbsp;|&nbsp; "
            f"5d trend: **{trend5:+.2%}** &nbsp;|&nbsp; "
            f"VIX: **{vix:.1f}** &nbsp;|&nbsp; "
            f"Conviction multiplier: **{mult:.2f}×**"
        )

    if "portfolio" in st.session_state:
        df = st.session_state["portfolio"]
        buys  = df[df["action"] == "BUY"]
        sells = df[df["action"] == "SELL"]

        c1, c2, c3 = st.columns(3)
        c1.metric("BUY candidates",  len(buys))
        c2.metric("SELL candidates", len(sells))
        c3.metric("Total stocks",    len(df))

        st.subheader("Full Ranking")
        display_cols = [c for c in [
            "ticker", "action", "gnn_rank", "sentiment_score",
            "final_score", "adjusted_score", "news_count",
        ] if c in df.columns]
        styled = df[display_cols].style.map(_color_action, subset=["action"])
        st.dataframe(styled, use_container_width=True, height=500)

        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "portfolio.csv", "text/csv")

        if "sentiment_score" in df.columns and not no_news:
            st.subheader("GNN Score vs News Sentiment")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            colors_map = {"BUY": "#2ecc71", "SELL": "#e74c3c", "HOLD": "#95a5a6"}
            for action, grp in df.groupby("action"):
                ax.scatter(grp["gnn_rank"], grp["sentiment_score"],
                           label=action, c=colors_map.get(action, "gray"), s=60, alpha=0.8)
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
                "Run":            r.get("folder", "?"),
                "Model":          r.get("source", "?").upper(),
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
                ax.set_title(metric)
                ax.set_xticklabels(sub["Run"], rotation=20, ha="right")
                ax.axhline(0, color="gray", lw=0.8, ls=":")
                ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

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
    all_tickers = tickers_file.read_text().strip().splitlines() if tickers_file.exists() else []

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
        st.markdown(f"**Overall Sentiment: :{color_map[sentiment_label]}[{sentiment_label}]**")

        if r["headlines"]:
            st.subheader("Headlines & FinBERT Scores")
            headlines_df = pd.DataFrame(r["headlines"])
            if not headlines_df.empty:
                def _sentiment_color(row):
                    if row["positive"] > row["negative"] and row["positive"] > row["neutral"]:
                        return ["background-color: #d4edda"] * len(row)
                    if row["negative"] > row["positive"] and row["negative"] > row["neutral"]:
                        return ["background-color: #f8d7da"] * len(row)
                    return [""] * len(row)
                st.dataframe(
                    headlines_df.style.apply(_sentiment_color, axis=1),
                    use_container_width=True,
                )
        else:
            st.info(f"No news found for {selected_ticker} via yfinance.")


# ── Tab 4: System Overview ───────────────────────────────────────────────────
with tab4:
    st.header("Multi-Agent System Architecture")

    st.subheader("LangGraph Pipeline")
    try:
        from graph import get_mermaid
        mermaid_src = get_mermaid()
        st.markdown(f"```mermaid\n{mermaid_src}\n```")
    except Exception as e:
        st.warning(f"Could not render LangGraph diagram: {e}")
        st.markdown("""
```
START → gnn_node → news_node → portfolio_node → risk_node → macro_node → report_node → END
               ↘ (no_news)  ↗
```
""")

    st.markdown("""
## Pipeline Nodes

| Node | Agent | Output |
|------|-------|--------|
| `gnn_node` | GNNAgent (selected variant) | `{ticker: score}` for ~200-500 NIFTY stocks |
| `news_node` | NewsAgent (yfinance + FinBERT) | `{ticker: {sentiment, headlines}}` |
| `portfolio_node` | PortfolioAgent | BUY/SELL/HOLD ranked DataFrame |
| `risk_node` | RiskAgent (yfinance) | `{ticker: {beta, vol_20d, risk_label, ...}}` |
| `macro_node` | MarketContextAgent (yfinance) | `{regime, VIX, multiplier}` |
| `report_node` | ReportWriterAgent (Groq/Gemini) | Markdown research note |

## Model Variants

| Variant | Architecture | Checkpoint pattern |
|---------|-------------|-------------------|
| `hybrid` | BiGRU + SparseMoE + Causal Hypergraph + HetGAT | `*hybrid_best.dat` |
| `mamba`  | Mamba SSM + SparseMoE + Hypergraph + HetGAT | `*mamba_moe_best.dat` |
| `thgnn`  | GRU + Positive/Negative GAT (base model) | `*icrank_best.dat` |

## Signal Fusion
```
final_score    = α × GNN_rank + (1−α) × sentiment_norm
adjusted_score = final_score × confidence_multiplier   (BUY only)
```

## Free LLM Providers for Report
- **Groq** — `llama3-70b-8192` — sign up free at console.groq.com → set `GROQ_API_KEY`
- **Gemini** — `gemini-1.5-flash` — sign up free at aistudio.google.com → set `GOOGLE_API_KEY`
""")

    wf_plot = ROOT / "THGNN" / "data" / "plots" / "walk_forward_results.png"
    if wf_plot.exists():
        st.subheader("Walk-Forward Validation Results")
        st.image(str(wf_plot), use_column_width=True)


# ── Tab 5: Risk Dashboard ────────────────────────────────────────────────────
with tab5:
    st.header("Risk Dashboard — BUY & SELL Candidates")

    if "risk_data" not in st.session_state or not st.session_state["risk_data"]:
        st.info("Run the pipeline in **Live Portfolio** to populate risk metrics.")
    else:
        risk_data = st.session_state["risk_data"]
        risk_df = (
            pd.DataFrame(risk_data)
            .T
            .reset_index()
            .rename(columns={"index": "ticker"})
        )

        # Merge action from portfolio
        if "portfolio" in st.session_state:
            portfolio = st.session_state["portfolio"][["ticker", "action", "final_score"]]
            risk_df = risk_df.merge(portfolio, on="ticker", how="left")

        def _risk_color(val: str) -> str:
            if val == "HIGH":    return "background-color: #f8d7da; color: #721c24; font-weight: bold"
            if val == "MEDIUM":  return "background-color: #fff3cd; color: #856404; font-weight: bold"
            if val == "LOW":     return "background-color: #d4edda; color: #155724; font-weight: bold"
            return ""

        display_cols = [c for c in [
            "ticker", "action", "risk_label", "beta", "vol_20d",
            "pct_from_52w_high", "pct_from_52w_low", "range_position",
            "current_price", "atr",
        ] if c in risk_df.columns]

        for col in ["beta", "vol_20d", "pct_from_52w_high", "pct_from_52w_low",
                    "range_position", "current_price", "atr"]:
            if col in risk_df.columns:
                risk_df[col] = pd.to_numeric(risk_df[col], errors="coerce")

        styled = risk_df[display_cols].style.map(_risk_color, subset=["risk_label"])
        st.dataframe(styled, use_container_width=True)

        # Beta vs volatility scatter
        plot_df = risk_df.dropna(subset=["beta", "vol_20d"])
        if not plot_df.empty:
            st.subheader("Beta vs 20-day Volatility")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            label_colors = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71", "UNKNOWN": "#aaa"}
            for label, grp in plot_df.groupby("risk_label"):
                ax.scatter(
                    grp["beta"].astype(float),
                    grp["vol_20d"].astype(float) * 100,
                    label=label,
                    c=label_colors.get(label, "gray"),
                    s=90, alpha=0.85, edgecolors="white", linewidths=0.5,
                )
                for _, row in grp.iterrows():
                    ax.annotate(row["ticker"].replace(".NS", ""),
                                (float(row["beta"]), float(row["vol_20d"]) * 100),
                                fontsize=7, ha="center", va="bottom")
            ax.axvline(1.0, color="gray", ls="--", lw=0.8, label="Beta=1 (market)")
            ax.set_xlabel("Beta vs NIFTY 50")
            ax.set_ylabel("20-day Realised Volatility (%)")
            ax.set_title("Risk Quadrant: BUY and SELL Candidates")
            ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)

        # Range position bar chart
        rng_df = risk_df.dropna(subset=["range_position"]).copy()
        if not rng_df.empty:
            st.subheader("52-Week Range Position (0 = at low, 1 = at high)")
            import matplotlib.pyplot as plt
            rng_df = rng_df.sort_values("range_position", ascending=False)
            colors_rng = ["#2ecc71" if v > 0.5 else "#e74c3c" for v in rng_df["range_position"].astype(float)]
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.bar(rng_df["ticker"].str.replace(".NS", ""), rng_df["range_position"].astype(float),
                    color=colors_rng, alpha=0.85)
            ax2.axhline(0.5, color="gray", ls="--", lw=0.8)
            ax2.set_ylabel("Range Position")
            ax2.set_xticklabels(rng_df["ticker"].str.replace(".NS", ""), rotation=30, ha="right")
            ax2.grid(True, alpha=0.3, axis="y")
            st.pyplot(fig2, use_container_width=True)


# ── Tab 6: Daily Report ──────────────────────────────────────────────────────
with tab6:
    st.header("Daily Research Report")

    report_md = st.session_state.get("report_markdown", "")
    run_date  = st.session_state.get("run_date", "")
    variant   = st.session_state.get("model_variant", "hybrid")

    if report_md:
        st.caption(f"Generated for {run_date} using model={variant}")
        st.markdown(report_md)
        st.download_button(
            "Download .md",
            report_md.encode("utf-8"),
            file_name=f"{run_date}_{variant}_report.md",
            mime="text/markdown",
        )
    else:
        # Try to load the most recent saved report
        reports_dir = ROOT / "reports"
        saved = sorted(glob.glob(str(reports_dir / "*_report.md")), reverse=True)
        if saved:
            last = Path(saved[0])
            st.caption(f"Showing last saved report: **{last.name}**")
            try:
                st.markdown(last.read_text(encoding="utf-8"))
            except Exception:
                st.warning("Could not read saved report.")
        else:
            st.info(
                "No report yet. Run the pipeline with a Groq or Gemini API key to generate one.\n\n"
                "- **Groq** (free): sign up at console.groq.com → set `GROQ_API_KEY` environment variable\n"
                "- **Gemini** (free): sign up at aistudio.google.com → set `GOOGLE_API_KEY` environment variable"
            )
