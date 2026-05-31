import { motion } from 'framer-motion'

export default function Toast({ msg, type }) {
  const ok = type === 'ok'
  const c = ok ? '#3DD68C' : '#F0616D'
  return (
    <motion.div
      initial={{ x: 60, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 60, opacity: 0 }}
      transition={{ duration: 0.16 }}
      style={{
        position: 'fixed', bottom: 18, right: 18, zIndex: 100,
        display: 'flex', alignItems: 'center', gap: 9,
        padding: '10px 14px', maxWidth: 320,
        background: '#0E0E10',
        border: '1px solid #232327',
        borderLeft: `2px solid ${c}`,
        borderRadius: 3,
        fontSize: 11.5, color: '#E4E4E7',
        fontFamily: 'JetBrains Mono, monospace',
      }}>
      <span className="num" style={{ color: c, fontSize: 12, fontWeight: 700 }}>{ok ? '✓' : '✕'}</span>
      {msg}
    </motion.div>
  )
}
