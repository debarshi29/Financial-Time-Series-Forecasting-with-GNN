import { motion } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useStore } from '../lib/store'
import { EmptyState } from './shared'

export default function ReportPanel() {
  const { report, runDate, variant } = useStore()

  if (!report) return (
    <EmptyState icon="📝" title="No report yet"
      desc={`Run the pipeline with an LLM key to generate a research note.<br><br>
        <strong>Groq</strong>: console.groq.com → set <code>GROQ_API_KEY</code><br>
        <strong>Gemini</strong>: aistudio.google.com → set <code>GOOGLE_API_KEY</code>`}
    />
  )

  const download = () => {
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([report], { type: 'text/markdown' }))
    a.download = `${runDate || 'report'}_${variant || 'gnn'}_report.md`
    a.click()
  }

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 20 }}>
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#EAEAFF', margin: 0, letterSpacing: '-.015em' }}>
            Daily Research Note
          </h2>
          {runDate && (
            <p style={{ fontSize: 11, color: '#282850', marginTop: 5, fontFamily: 'JetBrains Mono, monospace' }}>
              {runDate}{variant && ` · ${variant}`}
            </p>
          )}
        </div>
        <button onClick={download} className="btn-ghost" style={{ marginTop: 2 }}>
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          Download .md
        </button>
      </div>

      <div className="card md-body" style={{ padding: '24px 28px' }}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
      </div>
    </motion.div>
  )
}
