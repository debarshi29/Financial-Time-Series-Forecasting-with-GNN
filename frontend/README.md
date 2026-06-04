# Frontend — Vite + React SPA

[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)

An alternative React-based SPA for the trading terminal dashboard. This frontend connects to the same `demo/api.py` REST endpoints as `demo/server.py` but provides a component-based architecture for more complex UI interactions.

> For the primary demo, use `demo/server.py` — it is self-contained and has no Node.js dependency.

---

## Quick Start

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

Ensure `demo/server.py` is running on port 8000 before starting the dev server — the Vite proxy forwards `/api/*` requests there.

### Production Build

```bash
npm run build
# Output in frontend/dist/ — serve with any static file server
```

---

## Project Structure

```
frontend/
├── src/
│   ├── components/         # Panel components (Portfolio, News, Risk, Report)
│   └── lib/                # API client + shared state management
├── public/                 # Static assets
├── index.html
├── vite.config.js          # Proxies /api/* to http://localhost:8000
├── tailwind.config.js
└── package.json
```
