import { useEffect, useState } from 'react'
import { useStore } from '../lib/store'

const TABS = ['Portfolio', 'Backtest', 'News', 'Overview', 'Risk', 'Report']

export default function Header() {
  const [time, setTime] = useState('')
  const { loading, variant, runDate, activeTab, setTab, portfolio } = useStore()

  useEffect(() => {
    const tick = () => setTime(
      new Date().toLocaleTimeString('en-IN', {
        hour12: false, timeZone: 'Asia/Kolkata',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      })
    )
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <header className="shrink-0 flex items-center h-[52px] px-5 bg-s1 z-30"
      style={{ borderBottom: '1px solid rgba(255,255,255,.055)' }}>

      {/* ── Logo ─────────────────────────────────── */}
      <div className="flex items-center gap-2.5 shrink-0">
        <div className="w-[28px] h-[28px] rounded-lg flex items-center justify-center shrink-0"
          style={{
            background: 'linear-gradient(145deg, #4F52D9, #6366F1)',
            boxShadow: '0 0 14px rgba(99,102,241,.5)',
          }}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
            stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="4" r="2"/>
            <circle cx="4" cy="20" r="2"/>
            <circle cx="20" cy="20" r="2"/>
            <line x1="12" y1="6" x2="4" y2="18"/>
            <line x1="12" y1="6" x2="20" y2="18"/>
            <line x1="5.5" y1="19" x2="18.5" y2="19"/>
          </svg>
        </div>
        <div>
          <div className="text-[13px] font-bold text-t0" style={{ letterSpacing: '-.015em' }}>
            GNN Quant
          </div>
          <div className="text-[9.5px] font-medium text-t3 tracking-wide">NIFTY 500</div>
        </div>
      </div>

      <div className="shrink-0 w-px h-5 bg-bds mx-5" />

      {/* ── Navigation — lives in the header ─────── */}
      <nav className="flex items-center gap-0.5 flex-1">
        {TABS.map((tab, i) => {
          const active = activeTab === i
          return (
            <button
              key={i}
              onClick={() => setTab(i)}
              className="relative flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12.5px] font-medium transition-all duration-150 select-none"
              style={{
                color: active ? '#C4C4FF' : '#505080',
                background: active ? 'rgba(99,102,241,.1)' : 'transparent',
              }}
              onMouseEnter={e => { if (!active) e.currentTarget.style.color = '#8080B8' }}
              onMouseLeave={e => { if (!active) e.currentTarget.style.color = '#505080' }}
            >
              {tab}
              {/* Active underline indicator */}
              {active && (
                <span className="nav-underline absolute bottom-[-1px] left-3 right-3 h-[2px] rounded-t-full"
                  style={{ background: '#6366F1' }} />
              )}
              {/* Portfolio count badge */}
              {i === 0 && portfolio.length > 0 && (
                <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full leading-none"
                  style={{
                    background: active ? 'rgba(99,102,241,.25)' : 'rgba(255,255,255,.05)',
                    color: active ? '#A5A8FF' : '#505080',
                  }}>
                  {portfolio.length}
                </span>
              )}
            </button>
          )
        })}
      </nav>

      {/* ── Right: run context + status + clock ──── */}
      <div className="flex items-center gap-4 shrink-0">

        {/* Last run info */}
        {runDate && (
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-t3">run</span>
            <span className="text-[10.5px] font-mono text-t1"
              style={{
                background: 'rgba(255,255,255,.04)',
                border: '1px solid rgba(255,255,255,.07)',
                padding: '2px 8px', borderRadius: 5,
              }}>
              {runDate}
            </span>
            {variant && (
              <span className="text-[10px] font-mono font-bold"
                style={{ color: '#818CF8', background: 'rgba(99,102,241,.1)', padding: '2px 7px', borderRadius: 5 }}>
                {variant}
              </span>
            )}
          </div>
        )}

        <div className="w-px h-4 bg-bds" />

        {/* Status */}
        <div className="flex items-center gap-2">
          <div className="relative flex items-center justify-center w-4 h-4">
            {loading && (
              <span className="absolute status-ping w-3 h-3 rounded-full"
                style={{ background: 'rgba(240,160,64,.3)' }} />
            )}
            <span className="w-[7px] h-[7px] rounded-full transition-colors duration-300"
              style={{ background: loading ? '#F0A040' : '#22D18E' }} />
          </div>
          <span className="text-[11px] font-medium"
            style={{ color: loading ? '#F0A040' : '#505080' }}>
            {loading ? 'Running' : 'Ready'}
          </span>
        </div>

        <div className="w-px h-4 bg-bds" />

        {/* Clock */}
        <span className="font-mono text-[10.5px] text-t3 tabular-nums">{time}</span>
      </div>
    </header>
  )
}
