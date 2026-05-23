"""
Starts the FastAPI API server with uvicorn.
The built React frontend is served from demo/static/.

Usage (from project root):
    python demo/serve.py
    python demo/serve.py --port 8000

For hot-reload during development, run these two in separate terminals:
    python demo/serve.py              (API on :8000)
    cd frontend && npm run dev        (React dev server on :5173 with /api proxy)
"""
import subprocess
import sys

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", default=False)
    args = p.parse_args()

    cmd = [
        sys.executable, "-m", "uvicorn",
        "demo.api:app",
        "--host", "0.0.0.0",
        "--port", str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")

    print(f"\n  GNN + News Demo  ·  API server")
    print(f"  ──────────────────────────────────────")
    print(f"  API:  http://localhost:{args.port}/api/...")
    if not args.reload:
        from pathlib import Path
        static = Path(__file__).parent / "static" / "index.html"
        if static.exists():
            print(f"  Note: serve the built frontend by opening demo/static/index.html")
            print(f"        or run 'cd frontend && npm run dev' for dev mode (:5173)")
    print()

    subprocess.run(cmd)
