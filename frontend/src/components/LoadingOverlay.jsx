import { motion, AnimatePresence } from 'framer-motion'
import { useStore } from '../lib/store'

const STEPS = [
  'Fetching OHLCV data from yfinance…',
  'Loading GNN checkpoint…',
  'Running graph attention forward pass…',
  'Scoring NIFTY 500 stocks…',
  'Fetching news headlines…',
  'Running FinBERT sentiment analysis…',
  'Fusing GNN + sentiment signals…',
  'Computing portfolio risk metrics…',
  'Generating AI research report…',
]

export default function LoadingOverlay() {
  const { loading, loadingMsg } = useStore()

  return (
    <AnimatePresence>
      {loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          style={{
            position: 'fixed', inset: 0, zIndex: 50,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(7,7,13,.92)', backdropFilter: 'blur(10px)',
          }}>
          <motion.div
            initial={{ scale: .92, opacity: 0, y: 16 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: .95, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.25, 0.46, 0.45, 0.94] }}
            style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 28, maxWidth: 320, width: '100%', padding: '0 24px' }}>

            {/* Spinner */}
            <div style={{ position: 'relative', width: 72, height: 72 }}>
              <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: '1px solid rgba(99,102,241,.12)', animation: 'ping 2s ease-out infinite' }} />
              <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: '2px solid rgba(99,102,241,.08)' }} />
              <svg style={{ position: 'absolute', inset: 0, animation: 'spin .85s linear infinite' }} viewBox="0 0 72 72">
                <circle cx="36" cy="36" r="32" fill="none" stroke="url(#lg)" strokeWidth="3"
                  strokeLinecap="round" strokeDasharray="140" strokeDashoffset="105" />
                <defs>
                  <linearGradient id="lg" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#6366F1"/>
                    <stop offset="100%" stopColor="#A87BFF"/>
                  </linearGradient>
                </defs>
              </svg>
              <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#6366F1', animation: 'pulse 2s ease-in-out infinite' }} />
              </div>
            </div>

            {/* Message */}
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: 14, fontWeight: 600, color: '#EAEAFF', marginBottom: 6 }}>
                {loadingMsg || 'Running pipeline…'}
              </p>
              <p style={{ fontSize: 11, color: '#282850' }}>30–120 s depending on model variant</p>
            </div>

            {/* Step log */}
            <div className="card" style={{ width: '100%', padding: '14px 16px', maxHeight: 140, overflow: 'hidden', position: 'relative' }}>
              {STEPS.map((s, i) => (
                <motion.p key={i}
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.26, duration: 0.18 }}
                  style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 10.5, fontFamily: 'JetBrains Mono, monospace', color: '#282850', marginBottom: 5 }}>
                  <span style={{ color: '#6366F1' }}>›</span>
                  {s}
                </motion.p>
              ))}
              <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 36, background: 'linear-gradient(transparent, #101022)', borderRadius: '0 0 12px 12px', pointerEvents: 'none' }} />
            </div>

            {/* Bounce dots */}
            <div style={{ display: 'flex', gap: 6 }}>
              {[0, 1, 2].map(i => (
                <motion.div key={i}
                  animate={{ y: [0, -7, 0] }}
                  transition={{ duration: .75, delay: i * .14, repeat: Infinity, ease: 'easeInOut' }}
                  style={{ width: 6, height: 6, borderRadius: '50%', background: '#6366F1', opacity: .7 }} />
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
