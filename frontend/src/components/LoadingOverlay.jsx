import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import { useStore } from '../lib/store'

const STEPS = [
  'fetch OHLCV · yfinance',
  'load GNN checkpoint',
  'graph attention forward pass',
  'score NIFTY 500',
  'fetch news headlines',
  'FinBERT sentiment',
  'fuse GNN + sentiment',
  'compute risk metrics',
  'generate research note',
]

export default function LoadingOverlay() {
  const { loading, loadingMsg } = useStore()
  const [step, setStep] = useState(0)

  useEffect(() => {
    if (!loading) { setStep(0); return }
    const id = setInterval(() => setStep(s => Math.min(s + 1, STEPS.length - 1)), 1400)
    return () => clearInterval(id)
  }, [loading])

  return (
    <AnimatePresence>
      {loading && (
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          transition={{ duration: 0.12 }}
          style={{ position: 'fixed', inset: 0, zIndex: 50, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(10,10,11,.9)', backdropFilter: 'blur(3px)' }}>
          <div className="panel" style={{ width: 340, padding: 0 }}>
            <div className="panel-head">
              <span className="panel-title flex items-center gap-2">
                <Spin /> {loadingMsg || 'RUNNING'}
              </span>
              <span className="num text-[10px] text-tx3">{step + 1}/{STEPS.length}</span>
            </div>
            <div style={{ padding: '12px 0' }}>
              {STEPS.map((s, i) => {
                const done = i < step, active = i === step
                return (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '4px 16px', opacity: i > step ? .3 : 1 }}>
                    <span className="num" style={{ fontSize: 11, color: done ? '#3DD68C' : active ? '#56B6C2' : '#3F3F46', width: 10 }}>
                      {done ? '✓' : active ? '›' : '·'}
                    </span>
                    <span className="num" style={{ fontSize: 11, color: active ? '#E4E4E7' : done ? '#52525B' : '#3F3F46' }}>{s}</span>
                  </div>
                )
              })}
            </div>
            <div style={{ borderTop: '1px solid #232327', padding: '8px 16px' }}>
              <span className="num" style={{ fontSize: 9.5, color: '#3F3F46', letterSpacing: '.06em' }}>30–120s · varies by model variant</span>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

const Spin = () => (
  <svg className="spin" width="11" height="11" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="#56B6C2" strokeWidth="3" strokeOpacity=".25"/>
    <path d="M12 2a10 10 0 0 1 10 10" stroke="#56B6C2" strokeWidth="3" strokeLinecap="round"/>
  </svg>
)
