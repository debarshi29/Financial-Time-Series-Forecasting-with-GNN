"""SQLite-backed cache for yfinance news to avoid repeat API calls."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


_DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "news_cache.db"


def _conn(db_path: Path = _DEFAULT_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute(
        "CREATE TABLE IF NOT EXISTS news_cache "
        "(ticker TEXT, fetched_at REAL, payload TEXT, "
        "PRIMARY KEY (ticker, fetched_at))"
    )
    con.commit()
    return con


def get_cached(ticker: str, max_age_s: float = 3600.0, db_path: Path = _DEFAULT_DB) -> list[dict] | None:
    con = _conn(db_path)
    cutoff = time.time() - max_age_s
    row = con.execute(
        "SELECT payload FROM news_cache WHERE ticker=? AND fetched_at>=? ORDER BY fetched_at DESC LIMIT 1",
        (ticker, cutoff),
    ).fetchone()
    con.close()
    return json.loads(row[0]) if row else None


def store(ticker: str, news_items: list[dict], db_path: Path = _DEFAULT_DB) -> None:
    con = _conn(db_path)
    con.execute(
        "INSERT OR REPLACE INTO news_cache (ticker, fetched_at, payload) VALUES (?,?,?)",
        (ticker, time.time(), json.dumps(news_items)),
    )
    con.commit()
    con.close()
