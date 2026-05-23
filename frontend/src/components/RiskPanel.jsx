import { motion } from 'framer-motion'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine, BarChart, Bar } from 'recharts'
import { useStore } from '../lib/store'
import { ActionBadge, Card, EmptyState, RiskBadge, fmt } from './shared'

export default function RiskPanel() {
  const { riskData, portfolio } = useStore()

  if (!riskData || !Object.keys(riskData).length) return (
    <EmptyState icon="🛡️" title="No risk data"
      desc="Run the pipeline to compute live risk metrics (beta, volatility, ATR, 52w range)." />
  )

  const rows = Object.entries(riskData).map(([ticker, r]) => {
    const pr = portfolio.find(p => p.ticker === ticker) || {}
    return { ticker, ...r, action: pr.action }
  })

  const rCol = { HIGH: '#F05060', MEDIUM: '#F0A040', LOW: '#22D18E', UNKNOWN: '#505080' }

  const scatterData = rows
    .filter(r => r.beta != null && r.vol_20d != null)
    .map(r => ({ x: +r.beta, y: +r.vol_20d * 100, risk: r.risk_label || 'UNKNOWN', name: r.ticker.replace('.NS', ''), action: r.action }))

  const rangeSorted = [...rows].filter(r => r.range_position != null).sort((a, b) => +b.range_position - +a.range_position)

  const ttStyle = { background: '#101022', border: 'none', borderRadius: 10, boxShadow: '0 0 0 1px rgba(255,255,255,.07)', fontSize: 11 }

  return (
    <div>
      <Card title="Risk Metrics">
        <div style={{ maxHeight: 380, overflowY: 'auto', overflowX: 'auto' }}>
          <table className="dt">
            <thead style={{ position: 'sticky', top: 0 }}>
              <tr>
                {['Ticker', 'Signal', 'Risk', 'Beta', 'Vol 20d', 'Δ 52w High', 'Δ 52w Low', 'Range Pos.', 'Price (₹)', 'ATR']
                  .map(h => <th key={h}>{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <motion.tr key={r.ticker}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: Math.min(i * 0.01, 0.3) }}>
                  <td className="font-mono font-bold" style={{ color: '#EAEAFF' }}>{r.ticker.replace('.NS', '')}</td>
                  <td><ActionBadge action={r.action} /></td>
                  <td><RiskBadge level={r.risk_label} /></td>
                  <td className="font-mono tabular-nums">{fmt(r.beta, 2)}</td>
                  <td className="font-mono tabular-nums">{r.vol_20d ? (r.vol_20d * 100).toFixed(1) + '%' : '—'}</td>
                  <td className="font-mono tabular-nums" style={{ color: '#F05060' }}>{r.pct_from_52w_high ? r.pct_from_52w_high.toFixed(1) + '%' : '—'}</td>
                  <td className="font-mono tabular-nums" style={{ color: '#22D18E' }}>{r.pct_from_52w_low ? '+' + r.pct_from_52w_low.toFixed(1) + '%' : '—'}</td>
                  <td className="font-mono tabular-nums">{r.range_position != null ? (+r.range_position * 100).toFixed(0) + '%' : '—'}</td>
                  <td className="font-mono tabular-nums">{r.current_price ? '₹' + (+r.current_price).toFixed(2) : '—'}</td>
                  <td className="font-mono tabular-nums" style={{ color: '#505080' }}>{fmt(r.atr, 2)}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Beta vs 20-day Volatility">
        <div style={{ padding: '4px 16px 16px' }}>
          <ResponsiveContainer width="100%" height={310}>
            <ScatterChart margin={{ top: 14, right: 20, bottom: 24, left: 14 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.04)" />
              <XAxis dataKey="x" name="Beta" tick={{ fill: '#282850', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                label={{ value: 'Beta vs NIFTY 50', position: 'insideBottom', offset: -14, fill: '#505080', fontSize: 11 }} />
              <YAxis dataKey="y" name="Vol %" tick={{ fill: '#282850', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                label={{ value: '20d Vol (%)', angle: -90, position: 'insideLeft', fill: '#505080', fontSize: 11 }} />
              <ReferenceLine x={1} stroke="rgba(99,102,241,.3)" strokeDasharray="4 2"
                label={{ value: 'β=1', fill: '#505080', fontSize: 9 }} />
              <Tooltip
                content={({ payload }) => {
                  if (!payload?.length) return null
                  const d = payload[0].payload
                  return (
                    <div className="card" style={{ padding: '12px 16px', fontSize: 12 }}>
                      <p className="font-mono font-bold mb-2" style={{ color: '#EAEAFF' }}>{d.name}</p>
                      <p style={{ color: '#505080' }}>Beta: <span className="font-mono" style={{ color: '#8080B8' }}>{d.x.toFixed(2)}</span></p>
                      <p style={{ color: '#505080' }}>Vol: <span className="font-mono" style={{ color: '#8080B8' }}>{d.y.toFixed(1)}%</span></p>
                      <div className="mt-2"><RiskBadge level={d.risk} /></div>
                    </div>
                  )
                }}
              />
              <Scatter data={scatterData} isAnimationActive={false}>
                {scatterData.map((d, i) => <Cell key={i} fill={rCol[d.risk]} fillOpacity={0.85} />)}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card title="52-Week Range Position">
        <div style={{ padding: '4px 16px 16px' }}>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart
              data={rangeSorted.map(r => ({ name: r.ticker.replace('.NS', ''), value: +(+r.range_position * 100).toFixed(1), rp: +r.range_position }))}
              margin={{ top: 8, right: 8, bottom: 40, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.04)" />
              <XAxis dataKey="name" tick={{ fill: '#282850', fontSize: 8, fontFamily: 'JetBrains Mono' }} angle={-35} textAnchor="end" />
              <YAxis domain={[0, 100]} tick={{ fill: '#282850', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                label={{ value: 'Range %', angle: -90, position: 'insideLeft', fill: '#505080', fontSize: 10 }} />
              <ReferenceLine y={50} stroke="rgba(99,102,241,.3)" strokeDasharray="4 2" />
              <Tooltip contentStyle={ttStyle} formatter={v => [v.toFixed(0) + '%', '52w Range']} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]} isAnimationActive>
                {rangeSorted.map((r, i) => <Cell key={i} fill={+r.range_position > .5 ? '#22D18E' : '#F05060'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  )
}
