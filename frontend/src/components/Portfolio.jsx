import { useStore } from '../lib/store'
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts'
import { ActionBadge, Panel, EmptyState, Strip, Th, Bar, GhostBtn, fmt } from './shared'

const AX = { fill: '#52525B', fontSize: 9, fontFamily: 'JetBrains Mono' }

export default function Portfolio() {
  const { portfolio, macro, hasRun, sortCol, sortAsc, setSort } = useStore()

  if (!hasRun) return (
    <EmptyState title="NO PORTFOLIO DATA"
      desc="Configure the model in the sidebar and execute <strong>RUN PIPELINE</strong>." />
  )

  const buys  = portfolio.filter(r => r.action === 'BUY')
  const sells = portfolio.filter(r => r.action === 'SELL')
  const holds = portfolio.length - buys.length - sells.length

  const sorted = [...portfolio].sort((a, b) => {
    const av = a[sortCol], bv = b[sortCol]
    if (av == null) return 1; if (bv == null) return -1
    return sortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1)
  })

  const hasSent = portfolio.some(r => r.sentiment_score != null)
  const maxGnn = Math.max(...portfolio.map(r => Math.abs(r.gnn_rank ?? 0)), .001)
  const regime = macro?.market_regime || '—'
  const regCol = { BULL: '#3DD68C', BEAR: '#F0616D', SIDEWAYS: '#E5A94E' }[regime] || '#71717A'

  const scatter = portfolio.map(r => ({
    x: r.gnn_rank ?? 0, y: r.sentiment_score ?? 0, action: r.action,
    name: (r.ticker || '').replace('.NS', ''),
  }))
  const dot = { BUY: '#3DD68C', SELL: '#F0616D', HOLD: '#52525B' }

  return (
    <div>
      <Strip items={[
        { k: 'Buy',      v: buys.length,  color: '#3DD68C' },
        { k: 'Sell',     v: sells.length, color: '#F0616D' },
        { k: 'Hold',     v: holds },
        { k: 'Universe', v: portfolio.length },
        { k: 'Regime',   v: regime, color: regCol },
        { k: 'Conviction', v: (macro?.confidence_multiplier ?? 1).toFixed(2) + '×', color: '#56B6C2' },
        ...(macro?.vix_level != null ? [{ k: 'VIX', v: macro.vix_level.toFixed(1) }] : []),
      ]} />

      <Panel title="Stock Rankings" badge={portfolio.length}
        action={
          <GhostBtn onClick={() => downloadCSV(portfolio)}>
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            CSV
          </GhostBtn>
        }>
        <div style={{ maxHeight: 460, overflow: 'auto' }}>
          <table className="dt">
            <thead>
              <tr>
                <th className="r" style={{ width: 38 }}>#</th>
                <Th col="ticker"          cur={sortCol} asc={sortAsc} onSort={setSort}>Ticker</Th>
                <th>Sig</th>
                <Th col="gnn_rank"        cur={sortCol} asc={sortAsc} onSort={setSort}>GNN</Th>
                <Th col="sentiment_score" cur={sortCol} asc={sortAsc} onSort={setSort} r>Sent</Th>
                <Th col="final_score"     cur={sortCol} asc={sortAsc} onSort={setSort} r>Final</Th>
                <Th col="adjusted_score"  cur={sortCol} asc={sortAsc} onSort={setSort} r>Adj</Th>
                <Th col="news_count"      cur={sortCol} asc={sortAsc} onSort={setSort} r>News</Th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((row, i) => (
                <tr key={row.ticker || i}>
                  <td className="r num" style={{ color: '#3F3F46' }}>{i + 1}</td>
                  <td className="num font-semibold" style={{ color: '#E4E4E7' }}>{(row.ticker || '').replace('.NS', '')}</td>
                  <td><ActionBadge action={row.action} /></td>
                  <td>
                    <span className="inline-flex items-center gap-2">
                      <Bar value={Math.abs(row.gnn_rank)} max={maxGnn} color={(row.gnn_rank ?? 0) >= 0 ? '#56B6C2' : '#F0616D'} />
                      <span className="num" style={{ color: '#A1A1AA' }}>{fmt(row.gnn_rank, 4)}</span>
                    </span>
                  </td>
                  <td className="r num" style={{ color: sentCol(row.sentiment_score) }}>{fmtSigned(row.sentiment_score, 3)}</td>
                  <td className="r num font-semibold" style={{ color: '#E4E4E7' }}>{fmt(row.final_score, 4)}</td>
                  <td className="r num" style={{ color: '#A1A1AA' }}>{fmt(row.adjusted_score, 4)}</td>
                  <td className="r num" style={{ color: '#52525B' }}>{row.news_count ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {hasSent && (
        <Panel title="GNN Score × News Sentiment">
          <div style={{ padding: '14px 14px 8px' }}>
            <ResponsiveContainer width="100%" height={290}>
              <ScatterChart margin={{ top: 8, right: 18, bottom: 22, left: 8 }}>
                <CartesianGrid stroke="#18181B" />
                <XAxis dataKey="x" tick={AX} stroke="#232327"
                  label={{ value: 'GNN RANK', position: 'insideBottom', offset: -10, fill: '#52525B', fontSize: 9, letterSpacing: 1 }} />
                <YAxis dataKey="y" tick={AX} stroke="#232327"
                  label={{ value: 'SENTIMENT', angle: -90, position: 'insideLeft', fill: '#52525B', fontSize: 9, letterSpacing: 1 }} />
                <ReferenceLine x={0} stroke="#2E2E33" />
                <ReferenceLine y={0} stroke="#2E2E33" />
                <Tooltip cursor={{ stroke: '#2E2E33' }} content={<ScatterTip />} />
                <Scatter data={scatter} isAnimationActive={false}>
                  {scatter.map((d, i) => <Cell key={i} fill={dot[d.action] || '#52525B'} />)}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}
    </div>
  )
}

function ScatterTip({ payload }) {
  if (!payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="panel num" style={{ padding: '8px 11px', fontSize: 11 }}>
      <div className="font-semibold mb-1.5" style={{ color: '#E4E4E7' }}>{d.name}</div>
      <div style={{ color: '#52525B' }}>GNN&nbsp;&nbsp;<span style={{ color: '#A1A1AA' }}>{d.x.toFixed(4)}</span></div>
      <div style={{ color: '#52525B' }}>SENT&nbsp;<span style={{ color: '#A1A1AA' }}>{d.y.toFixed(3)}</span></div>
      <div className="mt-1.5"><ActionBadge action={d.action} /></div>
    </div>
  )
}

const sentCol = v => v == null ? '#52525B' : v > .05 ? '#3DD68C' : v < -.05 ? '#F0616D' : '#A1A1AA'
const fmtSigned = (v, d) => v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(d)

function downloadCSV(rows) {
  if (!rows.length) return
  const keys = Object.keys(rows[0])
  const csv = [keys.join(','), ...rows.map(r => keys.map(k => JSON.stringify(r[k] ?? '')).join(','))].join('\n')
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
  a.download = 'portfolio.csv'; a.click()
}
