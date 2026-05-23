// In dev, Vite proxies /api → localhost:8000. In prod, same origin.
const BASE = ''

async function post(path, body) {
  const r = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return r.json()
}

async function get(path) {
  const r = await fetch(BASE + path)
  return r.json()
}

export const api = {
  run:          (payload) => post('/api/run', payload),
  backtest:     ()        => get('/api/backtest'),
  tickers:      ()        => get('/api/tickers'),
  news:         (payload) => post('/api/news', payload),
  latestReport: ()        => get('/api/reports/latest'),
}
