"""
Report Writer Agent — generates a structured markdown research note using an LLM.

Supported providers:
    "custom" — any OpenAI-compatible chat endpoint. Set LLM_BASE_URL, LLM_API_KEY,
               and LLM_MODEL (the model identifier the endpoint expects).
    "groq"   — llama3-70b-8192 via Groq API  (set GROQ_API_KEY)
               Sign up free at https://console.groq.com
    "gemini" — gemini-1.5-flash via Google AI (set GOOGLE_API_KEY)
               Sign up free at https://aistudio.google.com

Uses LangChain chat model wrappers so provider is swappable with one parameter.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

SYSTEM_PROMPT = """You are a quantitative equity research analyst covering Indian stock markets (NSE/NIFTY universe).
Write a concise daily research note in markdown. Use exactly these six sections, in order:
## Executive Summary
## Top BUY Candidates
## SELL / SHORT Candidates
## Risk Warnings
## Market Context
## Disclaimer
Keep the total length under 600 words. Use bullet points within sections. Be direct and data-driven."""


def _build_llm(provider: str):
    """Return a LangChain chat model for the chosen provider."""
    if provider == "custom":
        from langchain_openai import ChatOpenAI
        base_url = os.environ.get("LLM_BASE_URL")
        api_key  = os.environ.get("LLM_API_KEY")
        model    = os.environ.get("LLM_MODEL")
        if not (base_url and api_key and model):
            raise EnvironmentError(
                "custom provider requires LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL."
            )
        return ChatOpenAI(
            model=model,
            temperature=0.3,
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama3-70b-8192", temperature=0.3)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Use 'custom', 'groq', or 'gemini'."
        )


def _auto_detect_provider() -> str | None:
    """Return the first provider whose API key is set in the environment."""
    if os.environ.get("LLM_API_KEY"):
        return "custom"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return None


def _build_prompt(
    date_str: str,
    portfolio_df: pd.DataFrame,
    news_signals: dict[str, dict],
    risk_data: dict[str, dict],
    macro_context: dict,
    model_variant: str,
    top_k: int,
) -> str:
    regime = macro_context.get("market_regime", "SIDEWAYS")
    nifty  = macro_context.get("nifty_return_1d", 0.0)
    vix    = macro_context.get("vix_level", 15.0)
    mult   = macro_context.get("confidence_multiplier", 1.0)

    buys  = portfolio_df[portfolio_df["action"] == "BUY"].head(top_k)
    sells = portfolio_df[portfolio_df["action"] == "SELL"].head(max(top_k // 2, 3))

    def _fmt_row(row: pd.Series) -> str:
        ticker = row["ticker"]
        score  = row.get("adjusted_score", row.get("final_score", 0.0))
        sent   = row.get("sentiment_score", 0.0)
        risk   = risk_data.get(ticker, {}).get("risk_label", "?")
        # Take top headline if available
        headlines = news_signals.get(ticker, {}).get("headlines", [])
        headline  = f'"{headlines[0]["title"]}"' if headlines else "no news"
        return f"  {ticker} | score={score:.3f} | sentiment={sent:+.3f} | risk={risk} | {headline}"

    buy_lines  = "\n".join(_fmt_row(r) for _, r in buys.iterrows())  if not buys.empty  else "  (none)"
    sell_lines = "\n".join(_fmt_row(r) for _, r in sells.iterrows()) if not sells.empty else "  (none)"

    return (
        f"Date: {date_str} | Model: {model_variant} | "
        f"Regime: {regime} | NIFTY 1d: {nifty:+.2%} | VIX: {vix:.1f} | "
        f"Conviction multiplier: {mult:.2f}\n\n"
        f"TOP BUY CANDIDATES (top {top_k}):\n{buy_lines}\n\n"
        f"SELL / SHORT CANDIDATES:\n{sell_lines}\n\n"
        "Write the research note now."
    )


class ReportWriterAgent:
    """Generate AI-written research reports using Groq or Gemini (both free)."""

    def __init__(self, provider: str | None = None) -> None:
        resolved = provider or _auto_detect_provider()
        if resolved is None:
            raise EnvironmentError(
                "No LLM API key found. Set LLM_API_KEY (custom), "
                "GROQ_API_KEY (groq), or GOOGLE_API_KEY (gemini)."
            )
        self.provider = resolved
        self._llm = _build_llm(resolved)
        print(f"[ReportAgent] Using provider={resolved}")

    def generate(
        self,
        date_str: str,
        portfolio_df: pd.DataFrame,
        news_signals: dict[str, dict],
        risk_data: dict[str, dict],
        macro_context: dict,
        model_variant: str = "hybrid",
        top_k: int = 5,
        save: bool = True,
    ) -> str:
        """Generate report synchronously. Returns the markdown string."""
        from langchain_core.messages import SystemMessage, HumanMessage

        human_text = _build_prompt(
            date_str, portfolio_df, news_signals, risk_data,
            macro_context, model_variant, top_k,
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_text)]
        response = self._llm.invoke(messages)
        markdown = response.content if hasattr(response, "content") else str(response)

        if save:
            self._save(date_str, model_variant, markdown)
        return markdown

    def generate_streaming(
        self,
        date_str: str,
        portfolio_df: pd.DataFrame,
        news_signals: dict[str, dict],
        risk_data: dict[str, dict],
        macro_context: dict,
        model_variant: str = "hybrid",
        top_k: int = 5,
        save: bool = True,
    ) -> Iterator[str]:
        """Generator that yields text chunks — use with Streamlit st.write_stream()."""
        from langchain_core.messages import SystemMessage, HumanMessage

        human_text = _build_prompt(
            date_str, portfolio_df, news_signals, risk_data,
            macro_context, model_variant, top_k,
        )
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_text)]
        full_text = []
        for chunk in self._llm.stream(messages):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_text.append(text)
            yield text

        if save:
            self._save(date_str, model_variant, "".join(full_text))

    def _save(self, date_str: str, model_variant: str, markdown: str) -> Path:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / f"{date_str}_{model_variant}_report.md"
        out.write_text(markdown, encoding="utf-8")
        print(f"[ReportAgent] Saved -> {out}")
        return out
