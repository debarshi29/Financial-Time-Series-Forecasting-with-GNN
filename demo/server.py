#!/usr/bin/env python3
"""
Lightweight demo server — zero extra dependencies (stdlib + what's already installed).
Run from the project root:
    python demo/server.py
    python demo/server.py --port 8080
"""
from __future__ import annotations

import glob
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MultiAgent"))
sys.path.insert(0, str(ROOT / "THGNN_MaGNet"))

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


def _get_template() -> str:
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def _load_backtest_results() -> list[dict]:
    results_dir = ROOT / "THGNN" / "data" / "backtest_results"
    out: list[dict] = []
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
            entry: dict = {"folder": folder.name, "source": "thgnn", "longshort": {}}
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


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default per-request logs
        pass

    def _send_json(self, data, status: int = 200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            self._send_html(_get_template())

        elif path == "/api/backtest":
            self._send_json(_load_backtest_results())

        elif path == "/api/tickers":
            f = ROOT / "THGNN" / "data" / "valid_nifty500.txt"
            tickers = f.read_text().strip().splitlines() if f.exists() else []
            self._send_json(tickers)

        elif path == "/api/reports/latest":
            reports_dir = ROOT / "reports"
            saved = sorted(glob.glob(str(reports_dir / "*_report.md")), reverse=True)
            if saved:
                content = Path(saved[0]).read_text(encoding="utf-8")
                self._send_json({"content": content, "filename": Path(saved[0]).name})
            else:
                self._send_json({"content": None, "filename": None})

        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/run":
            payload = self._read_body()
            try:
                from graph import run_graph_full  # type: ignore
                result = run_graph_full(
                    date=payload.get("date"),
                    top_k=payload.get("top_k", 10),
                    alpha=payload.get("alpha", 0.7),
                    no_news=payload.get("no_news", False),
                    no_report=payload.get("no_report", False),
                    model_variant=payload.get("model_variant", "hybrid"),
                    llm_provider=payload.get("llm_provider"),
                )
                self._send_json({
                    "success": True,
                    "portfolio":       result.get("portfolio_rows", []),
                    "risk_data":       result.get("risk_data", {}),
                    "macro_context":   result.get("macro_context", {}),
                    "report_markdown": result.get("report_markdown", ""),
                    "news_signals":    result.get("news_signals", {}),
                })
            except Exception as exc:
                self._send_json({"success": False, "error": str(exc)}, 500)

        elif path == "/api/news":
            payload = self._read_body()
            ticker = payload.get("ticker")
            max_h  = payload.get("max_headlines", 15)
            try:
                from utils.finbert_loader import get_finbert  # type: ignore
                get_finbert()
                from agents.news_agent import score_ticker  # type: ignore
                result = score_ticker(ticker, max_headlines=max_h, cache_max_age_s=600)
                self._send_json({"success": True, **result})
            except Exception as exc:
                self._send_json({"success": False, "error": str(exc)}, 500)

        else:
            self._send_json({"error": "Not found"}, 404)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GNN+News Demo Server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\n  GNN + News Forecasting  ·  Demo Server")
    print(f"  ─────────────────────────────────────────")
    print(f"  http://localhost:{args.port}")
    print(f"  Ctrl-C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")


if __name__ == "__main__":
    main()
