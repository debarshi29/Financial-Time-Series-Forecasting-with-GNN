import { motion } from 'framer-motion'

// ── Empty state ────────────────────────────────────────────────────────────────
export function EmptyState({ icon, title, desc }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center py-28 text-center select-none"
    >
      <div className="text-[32px] mb-5 opacity-60">{icon}</div>
      <p className="text-[14px] font-semibold mb-2" style={{ color: '#8080B8' }}>{title}</p>
      <p className="text-[12px] leading-relaxed max-w-xs" style={{ color: '#505080' }}
        dangerouslySetInnerHTML={{ __html: desc }} />
    </motion.div>
  )
}

// ── Metric card ────────────────────────────────────────────────────────────────
export function MetricCard({ label, value, color, sub }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="card p-5"
    >
      <div className="label-xs mb-3">{label}</div>
      <div className="font-mono font-bold tabular-nums leading-none"
        style={{ fontSize: 30, color: color || '#EAEAFF' }}>
        {value}
      </div>
      {sub && (
        <div className="text-[10.5px] mt-2.5" style={{ color: '#282850' }}>{sub}</div>
      )}
    </motion.div>
  )
}

// ── Card container ─────────────────────────────────────────────────────────────
export function Card({ children, title, action, className = '' }) {
  return (
    <div className={`card mb-5 ${className}`}>
      {(title || action) && (
        <div className="card-header">
          {title && <span className="card-header-title">{title}</span>}
          {action}
        </div>
      )}
      {children}
    </div>
  )
}

// ── Action badge ───────────────────────────────────────────────────────────────
export function ActionBadge({ action }) {
  const cfg = {
    BUY:  { bg: 'rgba(34,209,142,.12)', border: 'rgba(34,209,142,.3)',  color: '#22D18E', glyph: '▲', cls: 'badge-buy' },
    SELL: { bg: 'rgba(240,80,96,.12)',   border: 'rgba(240,80,96,.3)',   color: '#F05060', glyph: '▼', cls: '' },
    HOLD: { bg: 'rgba(255,255,255,.05)', border: 'rgba(255,255,255,.08)', color: '#505080', glyph: '●', cls: '' },
  }
  const c = cfg[action] || cfg.HOLD
  return (
    <span className={`inline-flex items-center gap-1.5 text-[10px] font-bold tracking-wider px-2.5 py-1 rounded-md ${c.cls}`}
      style={{ background: c.bg, border: `1px solid ${c.border}`, color: c.color }}>
      <span style={{ fontSize: 8 }}>{c.glyph}</span>
      {action || 'HOLD'}
    </span>
  )
}

// ── Sortable table header ──────────────────────────────────────────────────────
export function SortableTh({ col, cur, asc, onSort, children, right }) {
  const active = cur === col
  return (
    <th onClick={() => onSort(col)} className={right ? 'text-right' : ''}>
      <span className="inline-flex items-center gap-1">
        {children}
        <span style={{ fontSize: 9, opacity: active ? 1 : .3 }}>
          {active ? (asc ? '↑' : '↓') : '↕'}
        </span>
      </span>
    </th>
  )
}

// ── Code badge ─────────────────────────────────────────────────────────────────
export function CodeBadge({ children, color = 'blue' }) {
  const s = {
    blue:   { bg: 'rgba(99,102,241,.12)',  border: 'rgba(99,102,241,.3)',  color: '#818CF8' },
    purple: { bg: 'rgba(168,123,255,.12)', border: 'rgba(168,123,255,.3)', color: '#A87BFF' },
    yellow: { bg: 'rgba(240,160,64,.12)',  border: 'rgba(240,160,64,.3)',  color: '#F0A040' },
    green:  { bg: 'rgba(34,209,142,.12)',  border: 'rgba(34,209,142,.3)',  color: '#22D18E' },
    cyan:   { bg: 'rgba(34,200,232,.12)',  border: 'rgba(34,200,232,.3)',  color: '#22C8E8' },
  }
  const t = s[color] || s.blue
  return (
    <span className="inline-flex items-center text-[10px] font-bold px-2 py-0.5 rounded-md font-mono"
      style={{ background: t.bg, border: `1px solid ${t.border}`, color: t.color }}>
      {children}
    </span>
  )
}

// ── Risk badge ─────────────────────────────────────────────────────────────────
export function RiskBadge({ level }) {
  const s = {
    HIGH:   { bg: 'rgba(240,80,96,.12)',  border: 'rgba(240,80,96,.3)',  color: '#F05060' },
    MEDIUM: { bg: 'rgba(240,160,64,.12)', border: 'rgba(240,160,64,.3)', color: '#F0A040' },
    LOW:    { bg: 'rgba(34,209,142,.12)', border: 'rgba(34,209,142,.3)', color: '#22D18E' },
  }
  const t = s[level]
  return (
    <span className="inline-flex items-center text-[10px] font-bold px-2 py-0.5 rounded-md"
      style={t
        ? { background: t.bg, border: `1px solid ${t.border}`, color: t.color }
        : { background: 'rgba(255,255,255,.04)', border: '1px solid rgba(255,255,255,.07)', color: '#505080' }
      }>
      {level || '—'}
    </span>
  )
}

// ── Score mini-bar ─────────────────────────────────────────────────────────────
export function ScoreBar({ value, max = 1, color = '#6366F1' }) {
  const pct = Math.min(Math.max((value ?? 0) / (max || 1), 0), 1) * 100
  return (
    <div className="score-track">
      <div className="score-fill" style={{ width: `${pct}%`, background: color }} />
    </div>
  )
}

// ── Ghost button ───────────────────────────────────────────────────────────────
export function GhostBtn({ onClick, children }) {
  return <button onClick={onClick} className="btn-ghost">{children}</button>
}

// ── Formatter ─────────────────────────────────────────────────────────────────
export function fmt(v, d = 2) {
  if (v == null || (typeof v === 'number' && isNaN(v))) return '—'
  return parseFloat(v).toFixed(d)
}
