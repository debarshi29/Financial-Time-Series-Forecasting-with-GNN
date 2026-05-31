import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useStore } from '../lib/store'
import { EmptyState, GhostBtn } from './shared'

export default function ReportPanel() {
  const { report, runDate, variant } = useStore()

  if (!report) return (
    <EmptyState title="NO REPORT"
      desc="Run with an LLM key to generate a research note.<br><code>GROQ_API_KEY</code> or <code>GOOGLE_API_KEY</code>." />
  )

  const download = () => {
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([report], { type: 'text/markdown' }))
    a.download = `${runDate || 'report'}_${variant || 'gnn'}_report.md`
    a.click()
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <div className="flex items-center gap-2.5">
          <span className="panel-title">Daily Research Note</span>
          {runDate && <span className="num text-[10px] text-tx3">{runDate}{variant && ` · ${variant}`}</span>}
        </div>
        <GhostBtn onClick={download}>
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          .MD
        </GhostBtn>
      </div>
      <div className="md-body" style={{ padding: '22px 26px' }}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
      </div>
    </div>
  )
}
