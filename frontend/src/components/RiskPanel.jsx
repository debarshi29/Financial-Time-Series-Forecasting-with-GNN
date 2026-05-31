import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine, BarChart, Bar } from 'recharts'
import { useStore } from '../lib/store'
import { ActionBadge, Panel, EmptyState, RiskBadge, fmt } from './shared'

const AX = { fill: '#52525B', fontSize: 9, fontFamily: 'JetBrains Mono' }

export default function RiskPanel() {
  const { riskData, portfolio } = useStore()

  if (!riskData || !Object.keys(riskData).length) return (
    <EmptyState title="NO RISK DATA"
      desc="Run the pipeline to compute live risk metrics (beta, volatility, ATR, 52w range)." />
  )

  const rows = Object.entries(riskData).map(([ticker, r]) => {
    const pr = portfolio.find(p => p.ticker === ticker) || {}
    return { ticker, ...r, action: pr.action }
  })

  const rCol = { HIGH: '#F0616D', MEDIUM: '#E5A94E', LOW: '#3DD68C', UNKNOWN: '#52525B' }

  const scatter = rows.filter(r => r.beta != null && r.vol_20d != null)
    .map(r => ({ x: +r.beta, y: +r.vol_20d * 100, risk: r.risk_label || 'UNKNOWN', name: r.ticker.replace('.NS', '') }))

  const range = [...rows].filter(r => r.range_position != null).sort((a, b) => +b.range_position - +a.range_position)

  return (
    <div>
      <Panel title="Risk Metrics" badge={rows.length}>
        <div style={{ maxHeight: 380, overflow: 'auto' }}>
          <table className="dt">
            <thead style={{ position: 'sticky', top: 0 }}>
              <tr>
                <th>Ticker</th><th>Sig</th><th>Risk</th>
                <th className="r">Beta</th><th className="r">Vol</th>
                <th className="r">52wH</th><th className="r">52wL</th>
                <th className="r">Range</th><th className="r">Price</th><th className="r">ATR</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(r => (
                <tr key={r.ticker}>
                  <td className="num font-semibold" style={{ color: '#E4E4E7' }}>{r.ticker.replace('.NS', '')}</td>
                  <td><ActionBadge action={r.action} /></td>
                  <td><RiskBadge level={r.risk_label} /></td>
                  <td className="r num">{fmt(r.beta, 2)}</td>
                  <td className="r num">{r.vol_20d ? (r.vol_20d * 100).toFixed(1) : '—'}</td>
                  <td className="r num" style={{ color: '#F0616D' }}>{r.pct_from_52w_high ? r.pct_from_52w_high.toFixed(1) : '—'}</td>
                  <td className="r num" style={{ color: '#3DD68C' }}>{r.pct_from_52w_low ? '+' + r.pct_from_52w_low.toFixed(1) : '—'}</td>
                  <td className="r num">{r.range_position != null ? (+r.range_position * 100).toFixed(0) : '—'}</td>
                  <td className="r num">{r.current_price ? (+r.current_price).toFixed(2) : '—'}</td>
                  <td className="r num" style={{ color: '#52525B' }}>{fmt(r.atr, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="Beta × 20d Volatility">
        <div style={{ padding: '14px 12px 8px' }}>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 8, right: 16, bottom: 22, left: 6 }}>
              <CartesianGrid stroke="#18181B" />
              <XAxis dataKey="x" tick={AX} stroke="#232327"
                label={{ value: 'BETA', position: 'insideBottom', offset: -10, fill: '#52525B', fontSize: 9, letterSpacing: 1 }} />
              <YAxis dataKey="y" tick={AX} stroke="#232327"
                label={{ value: 'VOL %', angle: -90, position: 'insideLeft', fill: '#52525B', fontSize: 9, letterSpacing: 1 }} />
              <ReferenceLine x={1} stroke="#56B6C2" strokeOpacity={.4} strokeDasharray="3 3" label={{ value: 'β=1', fill: '#52525B', fontSize: 8 }} />
              <Tooltip cursor={{ stroke: '#2E2E33' }} content={<RiskTip />} />
              <Scatter data={scatter} isAnimationActive={false}>
                {scatter.map((d, i) => <Cell key={i} fill={rCol[d.risk]} />)}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <Panel title="52-Week Range Position">
        <div style={{ padding: '14px 12px 8px' }}>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={range.map(r => ({ name: r.ticker.replace('.NS', ''), value: +(+r.range_position * 100).toFixed(1), rp: +r.range_position }))}
              margin={{ top: 6, right: 6, bottom: 36, left: 4 }}>
              <CartesianGrid stroke="#18181B" vertical={false} />
              <XAxis dataKey="name" tick={{ ...AX, fontSize: 8 }} stroke="#232327" angle={-35} textAnchor="end" />
              <YAxis domain={[0, 100]} tick={AX} stroke="#232327" />
              <ReferenceLine y={50} stroke="#56B6C2" strokeOpacity={.4} strokeDasharray="3 3" />
              <Tooltip cursor={{ fill: '#151517' }}
                contentStyle={{ background: '#0E0E10', border: '1px solid #232327', borderRadius: 3, fontSize: 11, fontFamily: 'JetBrains Mono' }}
                formatter={v => [v.toFixed(0) + '%', 'Range']} />
              <Bar dataKey="value" isAnimationActive={false}>
                {range.map((r, i) => <Cell key={i} fill={+r.range_position > .5 ? '#3DD68C' : '#F0616D'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Panel>
    </div>
  )
}

function RiskTip({ payload }) {
  if (!payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="panel num" style={{ padding: '8px 11px', fontSize: 11 }}>
      <div className="font-semibold mb-1.5" style={{ color: '#E4E4E7' }}>{d.name}</div>
      <div style={{ color: '#52525B' }}>BETA&nbsp;<span style={{ color: '#A1A1AA' }}>{d.x.toFixed(2)}</span></div>
      <div style={{ color: '#52525B' }}>VOL&nbsp;&nbsp;<span style={{ color: '#A1A1AA' }}>{d.y.toFixed(1)}%</span></div>
      <div className="mt-1.5"><RiskBadge level={d.risk} /></div>
    </div>
  )
}
