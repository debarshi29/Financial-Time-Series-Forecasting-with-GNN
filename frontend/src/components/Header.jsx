import { useEffect, useState } from 'react'
import { useStore } from '../lib/store'

const TABS = ['PORTFOLIO', 'BACKTEST', 'NEWS', 'OVERVIEW', 'RISK', 'REPORT']

export default function Header() {
  const [time, setTime] = useState('')
  const { loading, variant, runDate, activeTab, setTab, portfolio } = useStore()

  useEffect(() => {
    const tick = () => setTime(
      new Date().toLocaleTimeString('en-GB', {
        hour12: false, timeZone: 'Asia/Kolkata',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      })
    )
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <header className="shrink-0 flex items-stretch h-[38px] bg-bg1 border-b border-ln z-30">

      {/* ── Brand ─────────────────────────────── */}
      <div className="flex items-center gap-2.5 px-4 border-r border-ln shrink-0">
        <span className="text-acc text-[13px] font-bold tracking-wider leading-none select-none">⬡</span>
        <span className="text-tx0 text-[12.5px] font-bold tracking-[.12em] leading-none">GNN&nbsp;QUANT</span>
      </div>

      {/* ── Nav ───────────────────────────────── */}
      <nav className="flex items-stretch">
        {TABS.map((tab, i) => {
          const active = activeTab === i
          return (
            <button
              key={i}
              onClick={() => setTab(i)}
              className="relative flex items-center gap-1.5 px-3.5 text-[10.5px] font-semibold tracking-[.1em] border-r border-ln transition-colors"
              style={{
                color: active ? '#56B6C2' : '#71717A',
                background: active ? '#151517' : 'transparent',
              }}
              onMouseEnter={e => { if (!active) e.currentTarget.style.color = '#A1A1AA' }}
              onMouseLeave={e => { if (!active) e.currentTarget.style.color = '#71717A' }}
            >
              {tab}
              {i === 0 && portfolio.length > 0 && (
                <span className="num text-[9px] px-1 leading-none py-0.5 rounded-sm"
                  style={{ background: active ? 'rgba(86,182,194,.15)' : '#232327', color: active ? '#56B6C2' : '#71717A' }}>
                  {portfolio.length}
                </span>
              )}
              {active && <span className="absolute left-0 right-0 bottom-0 h-[2px]" style={{ background: '#56B6C2' }} />}
            </button>
          )
        })}
      </nav>

      <div className="flex-1" />

      {/* ── Right cluster ─────────────────────── */}
      <div className="flex items-stretch">
        {runDate && (
          <div className="flex items-center gap-2 px-4 border-l border-ln">
            <span className="lbl" style={{ letterSpacing: '.1em' }}>RUN</span>
            <span className="num text-[11px] text-tx1">{runDate}</span>
            {variant && (
              <span className="text-[10px] font-semibold tracking-wide px-1.5 py-0.5 rounded-sm"
                style={{ background: 'rgba(86,182,194,.1)', color: '#56B6C2' }}>
                {variant.toUpperCase()}
              </span>
            )}
          </div>
        )}

        {/* Status */}
        <div className="flex items-center gap-2 px-4 border-l border-ln">
          <span className={`w-[7px] h-[7px] rounded-full ${loading ? 'blink' : ''}`}
            style={{ background: loading ? '#E5A94E' : '#3DD68C' }} />
          <span className="text-[10px] font-semibold tracking-[.1em]"
            style={{ color: loading ? '#E5A94E' : '#71717A' }}>
            {loading ? 'RUNNING' : 'READY'}
          </span>
        </div>

        {/* Clock */}
        <div className="flex items-center px-4 border-l border-ln">
          <span className="num text-[11px] text-tx2 tracking-wide">{time}</span>
          <span className="text-[9px] text-tx4 ml-1.5 tracking-wider">IST</span>
        </div>
      </div>
    </header>
  )
}
