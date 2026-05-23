import { motion } from 'framer-motion'
import { useStore } from '../lib/store'
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts'
import { ActionBadge, Card, EmptyState, MetricCard, SortableTh, ScoreBar, GhostBtn, fmt } from './shared'

export default function Portfolio() {
  const { portfolio, macro, hasRun, sortCol, sortAsc, setSort } = useStore()

  if (!hasRun) return (
    <EmptyState icon="📊" title="No portfolio data yet"
      desc="Configure the model in the sidebar and click <strong>Run Pipeline</strong>." />
  )

  const buys  = portfolio.filter(r => r.action === 'BUY')
  const sells = portfolio.filter(r => r.action === 'SELL')

  const sorted = [...portfolio].sort((a, b) => {
    const av = a[sortCol], bv = b[sortCol]
    if (av == null) return 1; if (bv == null) return -1
    return sortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1)
  })

  const hasSentiment = portfolio.some(r => r.sentiment_score != null)
  const maxGnn = Math.max(...portfolio.map(r => Math.abs(r.gnn_rank ?? 0)), .001)

  const scatterData = portfolio.map(r => ({
    x: r.gnn_rank ?? 0,
    y: r.sentiment_score ?? 0,
    action: r.action,
    name: (r.ticker || '').replace('.NS', ''),
  }))
  const dotCol = { BUY: '#22D18E', SELL: '#F05060', HOLD: '#505080' }

  return (
    <div>
      {macro && Object.keys(macro).length > 0 && <RegimeBanner macro={macro} />}

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        <MetricCard label="BUY Signals"    value={buys.length}
          color="#22D18E" sub={`${pct(buys.length, portfolio.length)}% of universe`} />
        <MetricCard label="SELL Signals"   value={sells.length}
          color="#F05060" sub={`${pct(sells.length, portfolio.length)}% of universe`} />
        <MetricCard label="Total Scored"   value={portfolio.length} sub="NIFTY 500 stocks" />
        <MetricCard label="Macro Factor"
          value={macro?.confidence_multiplier ? macro.confidence_multiplier.toFixed(2) + '×' : '1.00×'}
          color="#818CF8" sub="regime-adjusted" />
      </div>

      <Card title="Stock Rankings"
        action={
          <GhostBtn onClick={() => downloadCSV(portfolio)}>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Export CSV
          </GhostBtn>
        }>
        <div style={{ maxHeight: 440, overflowY: 'auto', overflowX: 'auto' }}>
          <table className="dt">
            <thead>
              <tr>
                <th style={{ width: 40, textAlign: 'right' }}>#</th>
                <SortableTh col="ticker"          cur={sortCol} asc={sortAsc} onSort={setSort}>Ticker</SortableTh>
                <th>Signal</th>
                <SortableTh col="gnn_rank"        cur={sortCol} asc={sortAsc} onSort={setSort}>GNN Score</SortableTh>
                <SortableTh col="sentiment_score" cur={sortCol} asc={sortAsc} onSort={setSort}>Sentiment</SortableTh>
                <SortableTh col="final_score"     cur={sortCol} asc={sortAsc} onSort={setSort}>Final</SortableTh>
                <SortableTh col="adjusted_score"  cur={sortCol} asc={sortAsc} onSort={setSort}>Adjusted</SortableTh>
                <SortableTh col="news_count" right cur={sortCol} asc={sortAsc} onSort={setSort}>News</SortableTh>
              </tr>
            </thead>
            <tbody>
              {sorted.map((row, i) => (
                <motion.tr key={row.ticker || i}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: Math.min(i * 0.009, 0.2) }}>
                  <td className="font-mono text-right" style={{ color: '#282850', fontSize: 11 }}>{i + 1}</td>
                  <td className="font-mono font-bold" style={{ color: '#EAEAFF' }}>
                    {(row.ticker || '').replace('.NS', '')}
                  </td>
                  <td><ActionBadge action={row.action} /></td>
                  <td>
                    <div className="flex items-center gap-2">
                      <ScoreBar value={row.gnn_rank} max={maxGnn} color="#6366F1" />
                      <span className="font-mono tabular-nums" style={{ color: '#8080B8', fontSize: 11 }}>
                        {fmt(row.gnn_rank, 4)}
                      </span>
                    </div>
                  </td>
                  <td className="font-mono tabular-nums" style={{ color: sentColor(row.sentiment_score) }}>
                    {fmt(row.sentiment_score, 3)}
                  </td>
                  <td className="font-mono tabular-nums font-semibold" style={{ color: '#EAEAFF' }}>
                    {fmt(row.final_score, 4)}
                  </td>
                  <td className="font-mono tabular-nums" style={{ color: '#8080B8' }}>
                    {fmt(row.adjusted_score, 4)}
                  </td>
                  <td className="font-mono tabular-nums text-right" style={{ color: '#505080' }}>
                    {row.news_count ?? '—'}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {hasSentiment && (
        <Card title="GNN Score vs News Sentiment">
          <div style={{ padding: '4px 16px 16px' }}>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ top: 16, right: 24, bottom: 28, left: 16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.04)" />
                <XAxis dataKey="x" name="GNN"
                  tick={{ fill: '#282850', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  label={{ value: 'GNN Rank Score', position: 'insideBottom', offset: -14, fill: '#505080', fontSize: 11 }} />
                <YAxis dataKey="y" name="Sentiment"
                  tick={{ fill: '#282850', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  label={{ value: 'News Sentiment', angle: -90, position: 'insideLeft', fill: '#505080', fontSize: 11 }} />
                <ReferenceLine x={0} stroke="rgba(255,255,255,.06)" />
                <ReferenceLine y={0} stroke="rgba(255,255,255,.06)" />
                <Tooltip
                  content={({ payload }) => {
                    if (!payload?.length) return null
                    const d = payload[0].payload
                    return (
                      <div className="card" style={{ padding: '12px 16px', fontSize: 12 }}>
                        <p className="font-mono font-bold mb-2" style={{ color: '#EAEAFF' }}>{d.name}</p>
                        <p style={{ color: '#505080' }}>GNN: <span className="font-mono" style={{ color: '#8080B8' }}>{d.x.toFixed(4)}</span></p>
                        <p style={{ color: '#505080' }}>Sentiment: <span className="font-mono" style={{ color: '#8080B8' }}>{d.y.toFixed(3)}</span></p>
                        <div className="mt-2"><ActionBadge action={d.action} /></div>
                      </div>
                    )
                  }}
                />
                <Scatter data={scatterData} isAnimationActive={false}>
                  {scatterData.map((d, i) => (
                    <Cell key={i} fill={dotCol[d.action] || '#505080'} fillOpacity={0.85} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}
    </div>
  )
}

function RegimeBanner({ macro }) {
  const regime = macro.market_regime || 'SIDEWAYS'
  const cfgs = {
    BULL:     { color: '#22D18E', bg: 'rgba(34,209,142,.06)',  border: 'rgba(34,209,142,.18)' },
    BEAR:     { color: '#F05060', bg: 'rgba(240,80,96,.06)',   border: 'rgba(240,80,96,.18)' },
    SIDEWAYS: { color: '#F0A040', bg: 'rgba(240,160,64,.06)',  border: 'rgba(240,160,64,.18)' },
  }
  const c = cfgs[regime] || cfgs.SIDEWAYS
  const p = v => v != null ? (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%' : '—'

  return (
    <div className="flex items-center gap-5 flex-wrap mb-6 px-5 py-3 rounded-xl"
      style={{ background: c.bg, border: `1px solid ${c.border}` }}>
      <span className="text-[10px] font-bold tracking-[.12em] uppercase font-mono px-2.5 py-1 rounded-lg"
        style={{ color: c.color, background: `${c.color}18`, border: `1px solid ${c.border}` }}>
        {regime}
      </span>
      {[
        ['NIFTY 1d', p(macro.nifty_return_1d)],
        ['5d trend', p(macro.nifty_trend_5d)],
        ['VIX', (macro.vix_level || 0).toFixed(1)],
        ['Conviction', (macro.confidence_multiplier || 1).toFixed(2) + '×'],
      ].map(([l, v]) => (
        <span key={l} className="text-[11px]" style={{ color: '#505080' }}>
          {l}: <strong className="font-mono" style={{ color: '#8080B8', fontWeight: 600 }}>{v}</strong>
        </span>
      ))}
    </div>
  )
}

const pct = (n, t) => t > 0 ? ((n / t) * 100).toFixed(0) : 0
const sentColor = v => v == null ? '#8080B8' : v > .05 ? '#22D18E' : v < -.05 ? '#F05060' : '#8080B8'

function downloadCSV(rows) {
  if (!rows.length) return
  const keys = Object.keys(rows[0])
  const csv = [keys.join(','), ...rows.map(r => keys.map(k => JSON.stringify(r[k] ?? '')).join(','))].join('\n')
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
  a.download = 'portfolio.csv'; a.click()
}
