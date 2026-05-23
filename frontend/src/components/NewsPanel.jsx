import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { api } from '../lib/api'
import { Card, MetricCard, EmptyState } from './shared'
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
    setLoading(true, `Fetching news for ${ticker.replace('.NS', '')}…`)
    try {
      const d = await api.news({ ticker, max_headlines: maxH })
      if (!d.success) throw new Error(d.error)
      setResult({ ...d, ticker })
    } catch (e) { console.error(e) }
    finally { setLoading(false) }
  }

  const sc  = result?.sentiment_score ?? 0
  const lbl = sc > .1 ? 'POSITIVE' : sc < -.1 ? 'NEGATIVE' : 'NEUTRAL'
  const col = sc > .1 ? '#22D18E'  : sc < -.1 ? '#F05060'  : '#F0A040'

  return (
    <div>
      {/* Controls */}
      <Card>
        <div style={{ padding: '16px 20px', display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'flex-end' }}>
          <div style={{ flex: 1, minWidth: 160 }}>
            <div className="label-xs mb-2">Ticker</div>
            <select value={ticker} onChange={e => setTicker(e.target.value)}
              className="fi fi-select" style={{ maxWidth: 260 }}>
              {tickers.map(t => <option key={t} value={t}>{t.replace('.NS', '')}</option>)}
            </select>
          </div>
          <div style={{ width: 180 }}>
            <div className="label-xs mb-2" style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Headlines</span>
              <span className="font-mono" style={{ color: '#EAEAFF', fontWeight: 700 }}>{maxH}</span>
            </div>
            <input type="range" min={5} max={30} value={maxH}
              onChange={e => setMaxH(+e.target.value)} className="fi-range" />
          </div>
          <motion.button
            whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
            onClick={fetchNews}
            className="btn-run h-9 px-5 rounded-lg font-semibold text-[13px]">
            Fetch &amp; Score
          </motion.button>
        </div>
      </Card>

      {!result ? (
        <EmptyState icon="📰" title="Select a stock"
          desc="Choose a ticker and click <strong>Fetch &amp; Score</strong> to run FinBERT sentiment analysis." />
      ) : (
        <AnimatePresence mode="wait">
          <motion.div key={result.ticker}
            initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-5">
              <MetricCard label="Sentiment Score"
                value={(sc >= 0 ? '+' : '') + sc.toFixed(3)} color={col} />
              <MetricCard label="Overall Tone" value={lbl} color={col} />
              <MetricCard label="Headlines" value={result.news_count || 0} />
              <MetricCard label="Ticker"
                value={result.ticker.replace('.NS', '')} color="#818CF8" />
            </div>

            <Card title="Sentiment Distribution">
              <div style={{ padding: '20px 24px' }}>
                <SentimentBar score={sc} />
              </div>
            </Card>

            <div className="space-y-2">
              {(result.headlines || []).map((h, i) => (
                <HeadlineCard key={i} h={h} index={i} />
              ))}
              {!(result.headlines || []).length && (
                <div className="card" style={{ padding: 16, fontSize: 12, color: '#505080' }}>
                  No headlines found for this ticker.
                </div>
              )}
            </div>
          </motion.div>
        </AnimatePresence>
      )}
    </div>
  )
}

function SentimentBar({ score }) {
  const pct = Math.max(0, Math.min(1, (score + 1) / 2))
  const col  = score > .1 ? '#22D18E' : score < -.1 ? '#F05060' : '#F0A040'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <span style={{ fontSize: 10, color: '#282850', width: 40, textAlign: 'right', flexShrink: 0 }}>−1</span>
      <div style={{ flex: 1, height: 10, borderRadius: 99, background: 'rgba(255,255,255,.05)', position: 'relative', overflow: 'hidden' }}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct * 100}%` }}
          transition={{ duration: 0.7, ease: [0.25, 0.46, 0.45, 0.94] }}
          style={{ height: '100%', borderRadius: 99, background: `linear-gradient(90deg, #F05060, ${col})` }}
        />
        <div style={{ position: 'absolute', top: 0, left: '50%', width: 1, height: '100%', background: 'rgba(255,255,255,.1)', transform: 'translateX(-50%)' }} />
      </div>
      <span style={{ fontSize: 10, color: '#282850', width: 40, flexShrink: 0 }}>+1</span>
    </div>
  )
}

function HeadlineCard({ h, index }) {
  const p = h.positive || 0, n = h.negative || 0, u = h.neutral || 0
  const tot = p + n + u || 1
  const dom = p > n ? 'POSITIVE' : n > p ? 'NEGATIVE' : 'NEUTRAL'
  const col = p > n ? '#22D18E'  : n > p ? '#F05060'  : '#505080'

  return (
    <motion.div
      className="card"
      style={{ padding: '16px 20px' }}
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.035, duration: 0.2 }}>
      <p style={{ fontSize: 12.5, color: '#EAEAFF', lineHeight: 1.6, marginBottom: 10 }}>{h.headline || ''}</p>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{ display: 'flex', height: 4, borderRadius: 99, overflow: 'hidden', width: 80, flexShrink: 0 }}>
          <div style={{ width: `${p/tot*100}%`, background: '#22D18E' }} />
          <div style={{ width: `${n/tot*100}%`, background: '#F05060' }} />
          <div style={{ width: `${u/tot*100}%`, background: 'rgba(255,255,255,.08)' }} />
        </div>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '.06em', color: col }}>{dom}</span>
        <span style={{ fontSize: 10, fontFamily: 'JetBrains Mono', color: '#282850' }}>
          P:{(p*100).toFixed(0)}% · N:{(n*100).toFixed(0)}%
        </span>
      </div>
    </motion.div>
  )
}
