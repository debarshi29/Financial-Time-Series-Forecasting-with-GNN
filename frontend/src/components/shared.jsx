// ── Empty state — no emoji, thin glyph + text ───────────────────────────────────
export function EmptyState({ title, desc }) {
  return (
    <div className="flex flex-col items-center justify-center py-28 text-center select-none">
      <div className="w-9 h-9 mb-5 flex items-center justify-center border border-ln2 rounded-sm">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#52525B" strokeWidth="1.5">
          <rect x="3" y="3" width="18" height="18" rx="1"/>
          <path d="M3 9h18M9 21V9" strokeWidth="1.2"/>
        </svg>
      </div>
      <p className="text-[12.5px] font-semibold text-tx1 mb-2 tracking-wide">{title}</p>
      <p className="text-[11px] text-tx3 leading-relaxed max-w-xs"
        dangerouslySetInnerHTML={{ __html: desc }} />
    </div>
  )
}

// ── Panel ────────────────────────────────────────────────────────────────────
export function Panel({ children, title, badge, action, className = '' }) {
  return (
    <div className={`panel mb-4 ${className}`}>
      {(title || action) && (
        <div className="panel-head">
          <div className="flex items-center gap-2.5">
            {title && <span className="panel-title">{title}</span>}
            {badge != null && <span className="num text-[10px] text-tx3">[{badge}]</span>}
          </div>
          {action}
        </div>
      )}
      {children}
    </div>
  )
}

// ── Status strip cell ────────────────────────────────────────────────────────
export function Strip({ items }) {
  return (
    <div className="strip mb-4">
      {items.map((it, i) => (
        <div key={i} className="strip-cell">
          <span className="strip-k">{it.k}</span>
          <span className="strip-v" style={it.color ? { color: it.color } : {}}>{it.v}</span>
        </div>
      ))}
    </div>
  )
}

// ── Action badge ───────────────────────────────────────────────────────────────
export function ActionBadge({ action }) {
  const cfg = {
    BUY:  { c: '#3DD68C', g: '▲' },
    SELL: { c: '#F0616D', g: '▼' },
    HOLD: { c: '#71717A', g: '■' },
  }
  const a = cfg[action] || cfg.HOLD
  return (
    <span className="badge num" style={{ color: a.c, borderColor: `${a.c}44`, background: `${a.c}14` }}>
      <span style={{ fontSize: 7 }}>{a.g}</span>{action || 'HOLD'}
    </span>
  )
}

// ── Code / variant badge ─────────────────────────────────────────────────────
export function CodeBadge({ children, color = 'acc' }) {
  const map = { acc: '#56B6C2', pos: '#3DD68C', neg: '#F0616D', warn: '#E5A94E', violet: '#A78BFA' }
  const c = map[color] || color
  return (
    <span className="badge num" style={{ color: c, borderColor: `${c}44`, background: `${c}14` }}>
      {children}
    </span>
  )
}

// ── Risk badge ─────────────────────────────────────────────────────────────────
export function RiskBadge({ level }) {
  const map = { HIGH: '#F0616D', MEDIUM: '#E5A94E', LOW: '#3DD68C' }
  const c = map[level] || '#71717A'
  return (
    <span className="badge num" style={{ color: c, borderColor: `${c}44`, background: `${c}14` }}>
      {level || '—'}
    </span>
  )
}

// ── Sortable header ────────────────────────────────────────────────────────────
export function Th({ col, cur, asc, onSort, children, r }) {
  const active = cur === col
  return (
    <th onClick={() => onSort(col)} className={r ? 'r' : ''}>
      <span className="inline-flex items-center gap-1">
        {children}
        <span style={{ fontSize: 8, color: active ? '#56B6C2' : '#3F3F46' }}>{active ? (asc ? '▲' : '▼') : '◆'}</span>
      </span>
    </th>
  )
}

// ── Inline score bar ───────────────────────────────────────────────────────────
export function Bar({ value, max = 1, color = '#56B6C2' }) {
  const pct = Math.min(Math.max((value ?? 0) / (max || 1), 0), 1) * 100
  return (
    <span style={{ display: 'inline-block', width: 40, height: 2, background: '#2E2E33', verticalAlign: 'middle' }}>
      <span style={{ display: 'block', height: '100%', width: `${pct}%`, background: color }} />
    </span>
  )
}

export function GhostBtn({ onClick, children }) {
  return <button onClick={onClick} className="btn-ghost">{children}</button>
}

export function fmt(v, d = 2) {
  if (v == null || (typeof v === 'number' && isNaN(v))) return '—'
  return parseFloat(v).toFixed(d)
}
