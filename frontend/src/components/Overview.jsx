import { Panel, CodeBadge } from './shared'

const NODES = [
  { id: 'gnn_node',       desc: 'Runs the selected GNN variant on ~500 NIFTY stocks, producing a rank score per ticker.' },
  { id: 'news_node',      desc: 'Fetches headlines via yfinance and scores each with FinBERT for per-stock sentiment.' },
  { id: 'portfolio_node', desc: 'Fuses GNN rank and sentiment: final = α·GNN + (1−α)·sentiment_norm' },
  { id: 'risk_node',      desc: 'Computes live beta, 20d volatility, 52-week range and ATR; labels HIGH/MEDIUM/LOW.' },
  { id: 'macro_node',     desc: 'Computes NIFTY regime, VIX level and a conviction multiplier.' },
  { id: 'report_node',    desc: 'Uses Groq or Gemini to write a daily research note in Markdown.' },
]

const MODELS = [
  { id: 'hybrid', color: 'acc',    arch: 'BiGRU + SparseMoE + Causal Hypergraph + HetGAT', ckpt: '*hybrid_best.dat' },
  { id: 'mamba',  color: 'violet', arch: 'Mamba SSM + SparseMoE + Hypergraph + HetGAT',    ckpt: '*mamba_moe_best.dat' },
  { id: 'thgnn',  color: 'pos',    arch: 'GRU + Positive/Negative GAT (base)',              ckpt: '*icrank_best.dat' },
]

const PIPE = ['START', 'gnn', 'news', 'portfolio', 'risk', 'macro', 'report', 'END']

export default function Overview() {
  return (
    <div>
      <Panel title="LangGraph Pipeline">
        <div style={{ padding: '16px 14px', display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6 }}>
          {PIPE.map((n, i) => {
            const term = n === 'START' || n === 'END'
            return (
              <span key={n} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <span className="num" style={{
                  fontSize: 10.5, fontWeight: 600, letterSpacing: '.06em',
                  padding: '4px 9px', borderRadius: 2,
                  color: term ? '#71717A' : '#56B6C2',
                  border: `1px solid ${term ? '#2E2E33' : 'rgba(86,182,194,.35)'}`,
                  background: term ? 'transparent' : 'rgba(86,182,194,.08)',
                }}>{n}</span>
                {i < PIPE.length - 1 && <span style={{ color: '#3F3F46', fontSize: 11 }}>→</span>}
              </span>
            )
          })}
        </div>
      </Panel>

      <Panel title="Agent Nodes">
        <div>
          {NODES.map((n, i) => (
            <div key={n.id} style={{ display: 'flex', gap: 14, padding: '11px 14px', borderBottom: i === NODES.length - 1 ? 'none' : '1px solid #18181B' }}>
              <span className="num shrink-0" style={{ fontSize: 11, fontWeight: 600, color: '#56B6C2', width: 110 }}>{n.id}</span>
              <span style={{ fontSize: 11.5, color: '#71717A', lineHeight: 1.5, fontFamily: 'Inter, sans-serif' }}>{n.desc}</span>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Model Variants">
        <div style={{ overflowX: 'auto' }}>
          <table className="dt">
            <thead><tr><th>Variant</th><th>Architecture</th><th>Checkpoint</th></tr></thead>
            <tbody>
              {MODELS.map(m => (
                <tr key={m.id}>
                  <td><CodeBadge color={m.color}>{m.id}</CodeBadge></td>
                  <td style={{ color: '#A1A1AA', fontFamily: 'Inter, sans-serif' }}>{m.arch}</td>
                  <td className="num" style={{ color: '#52525B' }}>{m.ckpt}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="Signal Fusion">
        <div style={{ padding: 14 }}>
          <div className="num" style={{ background: '#0A0A0B', border: '1px solid #18181B', borderRadius: 3, padding: '16px 18px', fontSize: 12.5, lineHeight: 2.3 }}>
            <div>
              <span style={{ color: '#56B6C2', fontWeight: 600 }}>final_score</span>
              <span style={{ color: '#3F3F46' }}>{'  = α · '}</span>
              <span style={{ color: '#3DD68C' }}>GNN_rank</span>
              <span style={{ color: '#3F3F46' }}>{' + (1−α) · '}</span>
              <span style={{ color: '#E5A94E' }}>sentiment_norm</span>
            </div>
            <div>
              <span style={{ color: '#56B6C2', fontWeight: 600 }}>adj_score</span>
              <span style={{ color: '#3F3F46' }}>{'   = final_score · '}</span>
              <span style={{ color: '#A78BFA' }}>conviction</span>
              <span style={{ color: '#3F3F46', fontSize: 10, marginLeft: 10 }}>// BUY only</span>
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-3">
            {[
              { n: 'GROQ', d: 'llama3-70b-8192 · console.groq.com', k: 'GROQ_API_KEY' },
              { n: 'GEMINI', d: 'gemini-1.5-flash · aistudio.google.com', k: 'GOOGLE_API_KEY' },
            ].map(p => (
              <div key={p.k} style={{ border: '1px solid #18181B', borderRadius: 3, padding: '11px 13px' }}>
                <div className="lbl mb-1.5">{p.n}</div>
                <div style={{ fontSize: 11, color: '#52525B', marginBottom: 5, fontFamily: 'Inter, sans-serif' }}>{p.d}</div>
                <code className="num" style={{ fontSize: 10, color: '#56B6C2' }}>{p.k}</code>
              </div>
            ))}
          </div>
        </div>
      </Panel>
    </div>
  )
}
