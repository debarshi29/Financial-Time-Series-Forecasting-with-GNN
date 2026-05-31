import { useState } from 'react'
import { useStore } from '../lib/store'
import { api } from '../lib/api'

const yesterday = () => {
  const d = new Date(); d.setDate(d.getDate() - 1)
  return d.toISOString().split('T')[0]
}

const MODELS = [
  { id: 'hybrid', name: 'HYBRID', sub: 'BiGRU·MoE·Hypergraph' },
  { id: 'mamba',  name: 'MAMBA',  sub: 'SSM·MoE·Hypergraph' },
  { id: 'thgnn',  name: 'THGNN',  sub: 'GRU·GAT (base)' },
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
    setLoading(true, 'INITIALISING PIPELINE')
    try {
      const data = await api.run({
        date, top_k: topK, alpha,
        no_news: noNews, no_report: noReport,
        model_variant: model, llm_provider: llm || null,
      })
      if (!data.success) throw new Error(data.error)
      setResult(data, date, model)
      showToast(`${(data.portfolio || []).length} stocks scored`, 'ok')
    } catch (e) {
      showToast(e.message, 'err')
    } finally {
      setLoading(false)
    }
  }

  return (
    <aside className="w-[228px] shrink-0 flex flex-col bg-bg1 border-r border-ln">

      {/* Header */}
      <div className="h-[34px] flex items-center px-4 border-b border-ln shrink-0">
        <span className="lbl">Configuration</span>
      </div>

      <div className="flex-1 overflow-y-auto">

        {/* Model — radio rows */}
        <Block label="Model">
          <div className="space-y-px">
            {MODELS.map(m => {
              const on = model === m.id
              return (
                <button key={m.id} onClick={() => setModel(m.id)}
                  className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-sm transition-colors text-left"
                  style={{ background: on ? '#151517' : 'transparent', border: `1px solid ${on ? '#2E2E33' : 'transparent'}` }}>
                  <span className="w-[10px] h-[10px] rounded-full shrink-0 border flex items-center justify-center"
                    style={{ borderColor: on ? '#56B6C2' : '#3F3F46' }}>
                    {on && <span className="w-[4px] h-[4px] rounded-full" style={{ background: '#56B6C2' }} />}
                  </span>
                  <span className="flex flex-col gap-0.5 min-w-0">
                    <span className="text-[11px] font-semibold tracking-wide leading-none"
                      style={{ color: on ? '#E4E4E7' : '#A1A1AA' }}>{m.name}</span>
                    <span className="text-[9px] leading-none truncate" style={{ color: on ? '#56B6C2' : '#52525B' }}>{m.sub}</span>
                  </span>
                </button>
              )
            })}
          </div>
        </Block>

        <Block label="Date">
          <input type="date" value={date} onChange={e => setDate(e.target.value)} className="fi num" />
        </Block>

        <Block label="Top-K" value={topK}>
          <input type="range" min={3} max={20} value={topK} onChange={e => setTopK(+e.target.value)} className="fi-range" />
          <Ticks lo="3" hi="20" />
        </Block>

        <Block label="GNN Weight α" value={alpha.toFixed(2)}>
          <input type="range" min={0} max={1} step={0.05} value={alpha} onChange={e => setAlpha(+e.target.value)} className="fi-range" />
          <Ticks lo="news" hi="gnn" />
        </Block>

        <Block label="Flags">
          <div className="space-y-2">
            <Check id="noNews"   checked={noNews}   onChange={setNoNews}   label="Skip news (faster)" />
            <Check id="noReport" checked={noReport} onChange={setNoReport} label="Skip AI report" />
          </div>
        </Block>

        <Block label="LLM Provider">
          <select value={llm} onChange={e => setLlm(e.target.value)} className="fi fi-sel">
            <option value="">Auto-detect</option>
            <option value="groq">Groq (free)</option>
            <option value="gemini">Gemini (free)</option>
          </select>
        </Block>
      </div>

      {/* Run */}
      <div className="p-3 border-t border-ln shrink-0">
        <button onClick={run} disabled={loading} className="btn w-full">
          {loading
            ? <><Spin />RUNNING</>
            : <><svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><path d="M5 3l14 9-14 9V3z"/></svg>RUN PIPELINE</>}
        </button>
      </div>
    </aside>
  )
}

function Block({ label, value, children }) {
  return (
    <div className="px-4 py-3.5 border-b border-ln">
      <div className="flex items-center justify-between mb-2.5">
        <span className="lbl">{label}</span>
        {value != null && <span className="num text-[11px] font-semibold text-tx0">{value}</span>}
      </div>
      {children}
    </div>
  )
}

function Ticks({ lo, hi }) {
  return (
    <div className="flex justify-between mt-2">
      <span className="text-[8.5px] tracking-wider text-tx4 uppercase">{lo}</span>
      <span className="text-[8.5px] tracking-wider text-tx4 uppercase">{hi}</span>
    </div>
  )
}

function Check({ id, checked, onChange, label }) {
  return (
    <label htmlFor={id} className="flex items-center gap-2.5 cursor-pointer group">
      <span className="w-[13px] h-[13px] rounded-sm border flex items-center justify-center shrink-0 transition-colors"
        style={{ borderColor: checked ? '#56B6C2' : '#3F3F46', background: checked ? 'rgba(86,182,194,.12)' : 'transparent' }}>
        {checked && (
          <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="#56B6C2" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
        )}
        <input id={id} type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} className="sr-only" />
      </span>
      <span className="text-[11px]" style={{ color: checked ? '#A1A1AA' : '#71717A' }}>{label}</span>
    </label>
  )
}

const Spin = () => (
  <svg className="spin" width="12" height="12" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity=".25"/>
    <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round"/>
  </svg>
)
