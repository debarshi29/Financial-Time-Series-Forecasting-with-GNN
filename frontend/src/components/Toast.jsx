import { motion } from 'framer-motion'

export default function Toast({ msg, type }) {
  const ok = type === 'ok'
  const accent = ok ? '#22D18E' : '#F05060'
  return (
    <motion.div
      initial={{ x: 100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 100, opacity: 0 }}
      transition={{ type: 'spring', stiffness: 340, damping: 32 }}
      style={{
        position: 'fixed', bottom: 24, right: 24, zIndex: 100,
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '12px 16px', maxWidth: 320,
        background: '#101022',
        boxShadow: `0 0 0 1px rgba(255,255,255,.07), 0 4px 24px rgba(0,0,0,.6), 0 0 0 1px ${accent}22`,
        borderRadius: 12,
        borderLeft: `3px solid ${accent}`,
        fontSize: 13, fontWeight: 500, color: '#EAEAFF',
      }}>
      <div style={{
        width: 20, height: 20, borderRadius: '50%', flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: `${accent}18`, color: accent, fontSize: 10, fontWeight: 700,
      }}>
        {ok ? '✓' : '✕'}
      </div>
      {msg}
    </motion.div>
  )
}
