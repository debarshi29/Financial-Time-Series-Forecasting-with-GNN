import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer, ReferenceLine } from 'recharts'
import { api } from '../lib/api'
import { Card, EmptyState, CodeBadge, fmt } from './shared'

export default function Backtest() {
  const [results, setResults] = useState(null)

  useEffect(() => {
    api.backtest().then(setResults).catch(() => setResults([]))
  }, [])

  if (results === null) return (
    <div className="space-y-3">
      {[1, 2, 3].map(i => <div key={i} className="skeleton h-14 rounded-xl" />)}
    </div>
  )

  if (!results.length) return (
    <EmptyState icon="📈" title="No backtest results"
      desc={`No results in <code>THGNN/data/backtest_results/</code>.<br>
        Run: <code>python THGNN/backtest.py --start-date 2025-01-01 --end-date 2026-02-28 --top-k 5</code>`} />
  )

  const rows = results.map(r => ({
    run:   r.folder || '?',
    model: (r.source || '?').toUpperCase(),
    tr:    r.longshort?.total_return_pct,
    ar:    r.longshort?.ann_return_pct,
    sh:    r.longshort?.sharpe,
    dd:    r.longshort?.max_drawdown_pct,
    wr:    r.longshort?.win_rate_pct,
    ic:    r.ic?.ic_mean,
  }))

  const barCol = v => v >= 0 ? '#22D18E' : '#F05060'

  return (
    <div>
      <Card title="Performance Metrics">
        <div style={{ overflowX: 'auto' }}>
          <table className="dt">
            <thead>
              <tr>
                {['Run', 'Model', 'Total Ret %', 'Ann Ret %', 'Sharpe', 'Max DD %', 'Win Rate %', 'IC Mean']
                  .map(h => <th key={h}>{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  <td className="font-mono" style={{ fontSize: 10.5, color: '#505080', maxWidth: 140 }}>
                    <span className="truncate block">{r.run}</span>
                  </td>
                  <td>
                    <CodeBadge color={r.model === 'HYBRID' ? 'blue' : r.model === 'MAMBA' ? 'purple' : 'green'}>
                      {r.model}
                    </CodeBadge>
                  </td>
                  <td className="font-mono tabular-nums font-semibold"
                    style={{ color: (r.tr ?? 0) >= 0 ? '#22D18E' : '#F05060' }}>
                    {fmt(r.tr, 2)}%
                  </td>
                  <td className="font-mono tabular-nums"
                    style={{ color: (r.ar ?? 0) >= 0 ? '#22D18E' : '#F05060' }}>
                    {fmt(r.ar, 2)}%
                  </td>
                  <td className="font-mono tabular-nums font-semibold"
                    style={{ color: (r.sh ?? 0) >= 1 ? '#22D18E' : (r.sh ?? 0) < 0 ? '#F05060' : '#8080B8' }}>
                    {fmt(r.sh, 3)}
                  </td>
                  <td className="font-mono tabular-nums" style={{ color: '#F05060' }}>{fmt(r.dd, 2)}%</td>
                  <td className="font-mono tabular-nums" style={{ color: '#8080B8' }}>{fmt(r.wr, 1)}%</td>
                  <td className="font-mono tabular-nums" style={{ color: '#818CF8' }}>{fmt(r.ic, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[
          { title: 'Sharpe Ratio', key: 'sh', ref: 1 },
          { title: 'Total Return %', key: 'tr', ref: 0, suffix: '%' },
        ].map(cfg => (
          <Card key={cfg.title} title={cfg.title}>
            <div style={{ padding: '4px 16px 16px' }}>
              <ResponsiveContainer width="100%" height={210}>
                <BarChart
                  data={rows.map(r => ({ name: r.run.slice(-12), value: r[cfg.key] ?? 0 }))}
                  margin={{ top: 8, right: 8, bottom: 32, left: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.04)" />
                  <XAxis dataKey="name" tick={{ fill: '#282850', fontSize: 9, fontFamily: 'JetBrains Mono' }}
                    angle={-20} textAnchor="end" />
                  <YAxis tick={{ fill: '#282850', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                  <ReferenceLine y={cfg.ref} stroke="rgba(99,102,241,.3)" strokeDasharray="4 2" />
                  <Tooltip
                    contentStyle={{ background: '#101022', border: 'none', borderRadius: 10, boxShadow: '0 0 0 1px rgba(255,255,255,.07)', fontSize: 11 }}
                    labelStyle={{ color: '#EAEAFF', fontWeight: 600 }}
                    formatter={v => [v.toFixed(3) + (cfg.suffix || ''), cfg.title]}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {rows.map((r, i) => <Cell key={i} fill={barCol(r[cfg.key] ?? 0)} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}
