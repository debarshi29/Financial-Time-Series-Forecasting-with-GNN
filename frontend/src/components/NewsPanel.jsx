import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { Panel, Strip, EmptyState } from './shared'
import { useStore } from '../lib/store'

export default function NewsPanel() {
  const [tickers, setTickers] = useState([])
  const [ticker,  setTicker]  = useState('')
  const [maxH,    setMaxH]    = useState(15)
  const [result,  setResult]  = useState(null)
  const { setLoading } = useStore()

  useEffect(() => {
    api.tickers().then(t => { setTickers(t); if (t.length) setTicker(t[0]) }).catch(() => {})
  }, [])

  const fetchNews = async () => {
    if (!ticker) return
    setLoading(true, `FETCHING NEWS · ${ticker.replace('.NS', '')}`)
    try {
      const d = await api.news({ ticker, max_headlines: maxH })
      if (!d.success) throw new Error(d.error)
      setResult({ ...d, ticker })
    } catch (e) { console.error(e) }
    finally { setLoading(false) }
  }

  const sc  = result?.sentiment_score ?? 0
  const lbl = sc > .1 ? 'POSITIVE' : sc < -.1 ? 'NEGATIVE' : 'NEUTRAL'
  const col = sc > .1 ? '#3DD68C' : sc < -.1 ? '#F0616D' : '#E5A94E'

  return (
    <div>
      <Panel title="News Query">
        <div style={{ padding: '14px', display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'flex-end' }}>
          <div style={{ flex: 1, minWidth: 160 }}>
            <div className="lbl mb-2">Ticker</div>
            <select value={ticker} onChange={e => setTicker(e.target.value)} className="fi fi-sel num" style={{ maxWidth: 240 }}>
              {tickers.map(t => <option key={t} value={t}>{t.replace('.NS', '')}</option>)}
            </select>
          </div>
          <div style={{ width: 170 }}>
            <div className="lbl mb-2 flex justify-between"><span>Headlines</span><span className="num text-tx0">{maxH}</span></div>
            <input type="range" min={5} max={30} value={maxH} onChange={e => setMaxH(+e.target.value)} className="fi-range" />
          </div>
          <button onClick={fetchNews} className="btn">FETCH &amp; SCORE</button>
        </div>
      </Panel>

      {!result ? (
        <EmptyState title="SELECT A TICKER"
          desc="Choose a ticker and run FinBERT sentiment scoring on its latest headlines." />
      ) : (
        <>
          <Strip items={[
            { k: 'Score', v: (sc >= 0 ? '+' : '') + sc.toFixed(3), color: col },
            { k: 'Tone', v: lbl, color: col },
            { k: 'Headlines', v: result.news_count || 0 },
            { k: 'Ticker', v: result.ticker.replace('.NS', ''), color: '#56B6C2' },
          ]} />

          <Panel title="Sentiment Scale">
            <div style={{ padding: '18px 20px' }}>
              <SentBar score={sc} />
            </div>
          </Panel>

          <Panel title="Headlines" badge={(result.headlines || []).length}>
            {(result.headlines || []).length ? (
              <div>
                {(result.headlines || []).map((h, i) => <HeadRow key={i} h={h} last={i === result.headlines.length - 1} />)}
              </div>
            ) : (
              <div style={{ padding: 16, fontSize: 11, color: '#52525B' }}>No headlines found.</div>
            )}
          </Panel>
        </>
      )}
    </div>
  )
}

function SentBar({ score }) {
  const pct = Math.max(0, Math.min(1, (score + 1) / 2)) * 100
  const col  = score > .1 ? '#3DD68C' : score < -.1 ? '#F0616D' : '#E5A94E'
  return (
    <div className="flex items-center gap-3">
      <span className="num text-[10px] text-tx4 w-8 text-right shrink-0">−1.0</span>
      <div style={{ flex: 1, height: 6, background: '#1C1C1F', position: 'relative' }}>
        <div style={{ position: 'absolute', left: '50%', top: 0, width: 1, height: '100%', background: '#2E2E33' }} />
        <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: `${pct}%`, background: col, transition: 'width .4s' }} />
        <div style={{ position: 'absolute', left: `${pct}%`, top: -3, width: 2, height: 12, background: col, transform: 'translateX(-50%)' }} />
      </div>
      <span className="num text-[10px] text-tx4 w-8 shrink-0">+1.0</span>
    </div>
  )
}

function HeadRow({ h, last }) {
  const p = h.positive || 0, n = h.negative || 0, u = h.neutral || 0
  const tot = p + n + u || 1
  const dom = p > n ? 'POS' : n > p ? 'NEG' : 'NEU'
  const col = p > n ? '#3DD68C' : n > p ? '#F0616D' : '#71717A'
  return (
    <div style={{ padding: '12px 14px', borderBottom: last ? 'none' : '1px solid #18181B', fontFamily: 'Inter, sans-serif' }}>
      <p style={{ fontSize: 12.5, color: '#D4D4D8', lineHeight: 1.55, marginBottom: 8 }}>{h.headline || ''}</p>
      <div className="flex items-center gap-3">
        <div style={{ display: 'flex', height: 3, width: 88, overflow: 'hidden', flexShrink: 0 }}>
          <div style={{ width: `${p/tot*100}%`, background: '#3DD68C' }} />
          <div style={{ width: `${n/tot*100}%`, background: '#F0616D' }} />
          <div style={{ width: `${u/tot*100}%`, background: '#2E2E33' }} />
        </div>
        <span className="num" style={{ fontSize: 9.5, fontWeight: 600, letterSpacing: '.08em', color: col }}>{dom}</span>
        <span className="num" style={{ fontSize: 9.5, color: '#3F3F46' }}>P{(p*100).toFixed(0)} N{(n*100).toFixed(0)} U{(u*100).toFixed(0)}</span>
      </div>
    </div>
  )
}
