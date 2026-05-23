import { useState } from 'react'
import { motion } from 'framer-motion'
import { useStore } from '../lib/store'
import { api } from '../lib/api'

const yesterday = () => {
  const d = new Date(); d.setDate(d.getDate() - 1)
  return d.toISOString().split('T')[0]
}

const MODELS = [
  { id: 'hybrid', name: 'Hybrid', sub: 'BiGRU · MoE · Hypergraph' },
  { id: 'mamba',  name: 'Mamba',  sub: 'SSM · MoE · Hypergraph' },
  { id: 'thgnn',  name: 'THGNN',  sub: 'GRU · GAT (base)' },
]

export default function Sidebar({ showToast }) {
  const { setLoading, setResult, loading } = useStore()

  const [date,     setDate]     = useState(yesterday())
  const [topK,     setTopK]     = useState(10)
  const [alpha,    setAlpha]    = useState(0.70)
  const [model,    setModel]    = useState('hybrid')
  const [noNews,   setNoNews]   = useState(false)
  const [noReport, setNoReport] = useState(false)
  const [llm,      setLlm]      = useState('')

  const run = async () => {
    setLoading(true, 'Initialising pipeline…')
    try {
      const data = await api.run({
        date, top_k: topK, alpha,
        no_news: noNews, no_report: noReport,
        model_variant: model,
        llm_provider: llm || null,
      })
      if (!data.success) throw new Error(data.error)
      setResult(data, date, model)
      showToast(`Done · ${(data.portfolio || []).length} stocks scored`, 'ok')
    } catch (e) {
      showToast(e.message, 'err')
    } finally {
      setLoading(false)
    }
  }

  return (
    <aside className="w-[240px] shrink-0 flex flex-col bg-s1"
      style={{ borderRight: '1px solid rgba(255,255,255,.055)' }}>

      {/* ── Scrollable form area ─────────────── */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5">

        {/* Model selector — buttons, not a dropdown */}
        <div>
          <div className="label-xs mb-2.5">Model</div>
          <div className="space-y-1.5">
            {MODELS.map(m => (
              <button
                key={m.id}
                onClick={() => setModel(m.id)}
                className="w-full text-left px-3 py-2.5 rounded-lg transition-all duration-150"
                style={{
                  background: model === m.id ? 'rgba(99,102,241,.12)' : 'rgba(255,255,255,.025)',
                  border: `1px solid ${model === m.id ? 'rgba(99,102,241,.35)' : 'rgba(255,255,255,.05)'}`,
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[12px] font-semibold"
                    style={{ color: model === m.id ? '#C4C4FF' : '#8080B8' }}>
                    {m.name}
                  </span>
                  {model === m.id && (
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none"
                      stroke="#6366F1" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  )}
                </div>
                <div className="text-[10px] mt-0.5" style={{ color: model === m.id ? '#6366F1' : '#282850' }}>
                  {m.sub}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Divider */}
        <div style={{ height: 1, background: 'rgba(255,255,255,.05)' }} />

        {/* Date */}
        <div>
          <div className="label-xs mb-2">Date</div>
          <input type="date" value={date} onChange={e => setDate(e.target.value)} className="fi" />
        </div>

        {/* Top-K */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="label-xs">Top-K</div>
            <span className="font-mono text-[11px] font-bold text-t0">{topK}</span>
          </div>
          <input type="range" min={3} max={20} value={topK}
            onChange={e => setTopK(+e.target.value)} className="fi-range" />
          <div className="flex justify-between mt-1.5">
            <span className="text-[9px] text-t3">3</span>
            <span className="text-[9px] text-t3">20</span>
          </div>
        </div>

        {/* Alpha */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="label-xs">GNN Weight α</div>
            <span className="font-mono text-[11px] font-bold text-t0">{alpha.toFixed(2)}</span>
          </div>
          <input type="range" min={0} max={1} step={0.05} value={alpha}
            onChange={e => setAlpha(+e.target.value)} className="fi-range" />
          <div className="flex justify-between mt-1.5">
            <span className="text-[9px] text-t3">news only</span>
            <span className="text-[9px] text-t3">GNN only</span>
          </div>
        </div>

        {/* Divider */}
        <div style={{ height: 1, background: 'rgba(255,255,255,.05)' }} />

        {/* Toggles */}
        <div className="space-y-3">
          <Toggle id="noNews"   checked={noNews}   onChange={setNoNews}
            label="Skip news (faster)" />
          <Toggle id="noReport" checked={noReport} onChange={setNoReport}
            label="Skip AI report" />
        </div>

        {/* LLM */}
        <div>
          <div className="label-xs mb-2">LLM Provider</div>
          <select value={llm} onChange={e => setLlm(e.target.value)} className="fi fi-select">
            <option value="">Auto-detect</option>
            <option value="groq">Groq (free)</option>
            <option value="gemini">Gemini (free)</option>
          </select>
        </div>

      </div>

      {/* ── Run button ───────────────────────── */}
      <div className="p-4" style={{ borderTop: '1px solid rgba(255,255,255,.05)' }}>
        <motion.button
          onClick={run}
          disabled={loading}
          whileHover={!loading ? { scale: 1.02 } : {}}
          whileTap={!loading  ? { scale: 0.97 } : {}}
          className="btn-run w-full h-11 rounded-xl font-bold text-[13.5px] flex items-center justify-center gap-2.5"
        >
          {loading ? <><SpinIcon />Running…</> : <><PlayIcon />Run Pipeline</>}
        </motion.button>
      </div>
    </aside>
  )
}

// ── Toggle switch ──────────────────────────────────────────────────────────────
function Toggle({ id, checked, onChange, label }) {
  return (
    <label htmlFor={id} className="flex items-center gap-3 cursor-pointer">
      <div className="relative shrink-0" style={{ width: 30, height: 17 }}>
        <input id={id} type="checkbox" checked={checked}
          onChange={e => onChange(e.target.checked)} className="sr-only" />
        <div style={{
          width: 30, height: 17, borderRadius: 99,
          background: checked ? '#6366F1' : 'rgba(255,255,255,.08)',
          border: `1px solid ${checked ? 'rgba(99,102,241,.6)' : 'rgba(255,255,255,.1)'}`,
          transition: 'background .2s, border-color .2s',
          position: 'relative',
        }}>
          <div style={{
            position: 'absolute',
            top: 2, left: checked ? 14 : 2,
            width: 11, height: 11,
            borderRadius: '50%',
            background: 'white',
            boxShadow: '0 1px 3px rgba(0,0,0,.4)',
            transition: 'left .2s',
          }} />
        </div>
      </div>
      <span className="text-[12px]" style={{ color: checked ? '#8080B8' : '#505080' }}>{label}</span>
    </label>
  )
}

// ── Icons ──────────────────────────────────────────────────────────────────────
const PlayIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
    <path d="M5 3l14 9-14 9V3z"/>
  </svg>
)

const SpinIcon = () => (
  <svg className="animate-spin" width="14" height="14" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity=".2"/>
    <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round"/>
  </svg>
)
