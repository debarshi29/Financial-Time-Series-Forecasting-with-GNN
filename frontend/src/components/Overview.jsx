import { motion } from 'framer-motion'
import { Card, CodeBadge } from './shared'

const NODES = [
  { id: 'gnn_node',       desc: 'Runs the selected GNN variant on ~500 NIFTY stocks, producing a rank score per ticker.' },
  { id: 'news_node',      desc: 'Fetches headlines via yfinance and scores each with FinBERT, returning per-stock sentiment.' },
  { id: 'portfolio_node', desc: 'Fuses GNN rank and sentiment: final = α·GNN + (1−α)·sentiment_norm' },
  { id: 'risk_node',      desc: 'Computes live beta, 20d volatility, 52-week range and ATR; labels stocks HIGH/MEDIUM/LOW.' },
  { id: 'macro_node',     desc: 'Computes NIFTY regime (BULL/BEAR/SIDEWAYS), VIX level and a conviction multiplier.' },
  { id: 'report_node',    desc: 'Uses Groq or Gemini to write a daily research note in Markdown.' },
]

const MODELS = [
  { id: 'hybrid', label: 'hybrid', color: 'blue',   arch: 'BiGRU + SparseMoE + Causal Hypergraph + HetGAT', ckpt: '*hybrid_best.dat' },
  { id: 'mamba',  label: 'mamba',  color: 'purple', arch: 'Mamba SSM + SparseMoE + Hypergraph + HetGAT',    ckpt: '*mamba_moe_best.dat' },
  { id: 'thgnn',  label: 'thgnn',  color: 'green',  arch: 'GRU + Positive/Negative GAT (base model)',        ckpt: '*icrank_best.dat' },
]

const PIPELINE = ['START', 'gnn_node', 'news_node', 'portfolio_node', 'risk_node', 'macro_node', 'report_node', 'END']

export default function Overview() {
  return (
    <div>
      <Card title="LangGraph Pipeline">
        <div style={{ padding: '16px 20px', display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8 }}>
          {PIPELINE.map((node, i) => (
            <motion.div key={node}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: 11, fontWeight: 600,
                padding: '5px 10px', borderRadius: 7,
                background: node === 'START' || node === 'END'
                  ? 'rgba(168,123,255,.12)' : 'rgba(99,102,241,.08)',
                border: `1px solid ${node === 'START' || node === 'END' ? 'rgba(168,123,255,.3)' : 'rgba(99,102,241,.2)'}`,
                color: node === 'START' || node === 'END' ? '#A87BFF' : '#818CF8',
              }}>
                {node}
              </span>
              {i < PIPELINE.length - 1 && (
                <span style={{ color: '#282850', fontSize: 12 }}>→</span>
              )}
            </motion.div>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-5">
        {NODES.map((n, i) => (
          <motion.div key={n.id} className="card" style={{ padding: '18px 20px' }}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.08 + i * 0.05 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 10 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#6366F1', flexShrink: 0 }} />
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, fontWeight: 700, color: '#818CF8', textTransform: 'uppercase', letterSpacing: '.08em' }}>
                {n.id}
              </span>
            </div>
            <p style={{ fontSize: 12, color: '#505080', lineHeight: 1.65 }}>{n.desc}</p>
          </motion.div>
        ))}
      </div>

      <Card title="Model Variants">
        <div style={{ overflowX: 'auto' }}>
          <table className="dt">
            <thead>
              <tr><th>Variant</th><th>Architecture</th><th>Checkpoint</th></tr>
            </thead>
            <tbody>
              {MODELS.map(m => (
                <tr key={m.id}>
                  <td><CodeBadge color={m.color}>{m.label}</CodeBadge></td>
                  <td style={{ color: '#8080B8' }}>{m.arch}</td>
                  <td style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10.5, color: '#505080' }}>{m.ckpt}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Signal Fusion Formula">
        <div style={{ padding: '4px 20px 20px' }}>
          <div style={{
            background: '#07070D', borderRadius: 10,
            border: '1px solid rgba(255,255,255,.05)',
            padding: '20px 24px',
            fontFamily: 'JetBrains Mono, monospace', fontSize: 13, lineHeight: 2.4,
          }}>
            <div>
              <span style={{ color: '#818CF8', fontWeight: 700 }}>final_score</span>
              <span style={{ color: '#282850' }}>    = α × </span>
              <span style={{ color: '#22D18E' }}>GNN_rank</span>
              <span style={{ color: '#282850' }}> + (1 − α) × </span>
              <span style={{ color: '#22C8E8' }}>sentiment_norm</span>
            </div>
            <div>
              <span style={{ color: '#818CF8', fontWeight: 700 }}>adjusted_score</span>
              <span style={{ color: '#282850' }}> = final_score × </span>
              <span style={{ color: '#F0A040' }}>confidence_multiplier</span>
              <span style={{ color: '#282850', fontSize: 10, marginLeft: 12 }}>← BUY only</span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4">
            {[
              { name: 'Groq (free)',   desc: 'llama3-70b-8192', key: 'GROQ_API_KEY',   url: 'console.groq.com' },
              { name: 'Gemini (free)', desc: 'gemini-1.5-flash', key: 'GOOGLE_API_KEY', url: 'aistudio.google.com' },
            ].map(p => (
              <div key={p.key} style={{
                background: 'rgba(255,255,255,.025)', borderRadius: 8,
                border: '1px solid rgba(255,255,255,.06)',
                padding: '14px 16px',
              }}>
                <div className="label-xs mb-2">{p.name}</div>
                <p style={{ fontSize: 12, color: '#505080', marginBottom: 6 }}>{p.desc} · {p.url}</p>
                <code style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10.5, color: '#818CF8' }}>{p.key}</code>
              </div>
            ))}
          </div>
        </div>
      </Card>
    </div>
  )
}
