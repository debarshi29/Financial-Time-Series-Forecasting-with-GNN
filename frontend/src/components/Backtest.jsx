import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer, ReferenceLine } from 'recharts'
import { api } from '../lib/api'
import { Panel, EmptyState, CodeBadge, fmt } from './shared'

const AX = { fill: '#52525B', fontSize: 9, fontFamily: 'JetBrains Mono' }

export default function Backtest() {
  const [results, setResults] = useState(null)

  useEffect(() => { api.backtest().then(setResults).catch(() => setResults([])) }, [])

  if (results === null) return (
    <div className="space-y-3">{[1, 2, 3].map(i => <div key={i} className="skeleton h-12" />)}</div>
  )

  if (!results.length) return (
    <EmptyState title="NO BACKTEST RESULTS"
      desc="No results in <code>THGNN/data/backtest_results/</code>.<br>Run <code>python THGNN/backtest.py</code>." />
  )

  const rows = results.map(r => ({
    run: r.folder || '?', model: (r.source || '?').toUpperCase(),
    tr: r.longshort?.total_return_pct, ar: r.longshort?.ann_return_pct,
    sh: r.longshort?.sharpe, dd: r.longshort?.max_drawdown_pct,
    wr: r.longshort?.win_rate_pct, ic: r.ic?.ic_mean,
  }))

  const bc = v => v >= 0 ? '#3DD68C' : '#F0616D'

  return (
    <div>
      <Panel title="Performance Metrics" badge={rows.length}>
        <div style={{ overflowX: 'auto' }}>
          <table className="dt">
            <thead>
              <tr>
                <th>Run</th><th>Model</th>
                <th className="r">Total %</th><th className="r">Ann %</th><th className="r">Sharpe</th>
                <th className="r">Max DD</th><th className="r">Win %</th><th className="r">IC</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  <td className="num" style={{ color: '#52525B', maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.run}</td>
                  <td><CodeBadge color={r.model === 'HYBRID' ? 'acc' : r.model === 'MAMBA' ? 'violet' : 'pos'}>{r.model}</CodeBadge></td>
                  <td className="r num font-semibold" style={{ color: (r.tr ?? 0) >= 0 ? '#3DD68C' : '#F0616D' }}>{fmt(r.tr, 2)}</td>
                  <td className="r num" style={{ color: (r.ar ?? 0) >= 0 ? '#3DD68C' : '#F0616D' }}>{fmt(r.ar, 2)}</td>
                  <td className="r num font-semibold" style={{ color: (r.sh ?? 0) >= 1 ? '#3DD68C' : (r.sh ?? 0) < 0 ? '#F0616D' : '#A1A1AA' }}>{fmt(r.sh, 3)}</td>
                  <td className="r num" style={{ color: '#F0616D' }}>{fmt(r.dd, 2)}</td>
                  <td className="r num" style={{ color: '#A1A1AA' }}>{fmt(r.wr, 1)}</td>
                  <td className="r num" style={{ color: '#56B6C2' }}>{fmt(r.ic, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[{ t: 'Sharpe Ratio', k: 'sh', ref: 1 }, { t: 'Total Return %', k: 'tr', ref: 0 }].map(c => (
          <Panel key={c.t} title={c.t}>
            <div style={{ padding: '14px 12px 8px' }}>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={rows.map(r => ({ name: r.run.slice(-10), value: r[c.k] ?? 0 }))} margin={{ top: 6, right: 6, bottom: 28, left: 4 }}>
                  <CartesianGrid stroke="#18181B" vertical={false} />
                  <XAxis dataKey="name" tick={AX} stroke="#232327" angle={-20} textAnchor="end" />
                  <YAxis tick={AX} stroke="#232327" />
                  <ReferenceLine y={c.ref} stroke="#56B6C2" strokeOpacity={.4} strokeDasharray="3 3" />
                  <Tooltip cursor={{ fill: '#151517' }}
                    contentStyle={{ background: '#0E0E10', border: '1px solid #232327', borderRadius: 3, fontSize: 11, fontFamily: 'JetBrains Mono' }}
                    labelStyle={{ color: '#E4E4E7' }} formatter={v => [v.toFixed(3), c.t]} />
                  <Bar dataKey="value" isAnimationActive={false}>
                    {rows.map((r, i) => <Cell key={i} fill={bc(r[c.k] ?? 0)} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        ))}
      </div>
    </div>
  )
}
