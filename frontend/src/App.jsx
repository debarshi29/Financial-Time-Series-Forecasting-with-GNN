import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { useStore } from './lib/store'
import { api } from './lib/api'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import Portfolio from './components/Portfolio'
import Backtest from './components/Backtest'
import NewsPanel from './components/NewsPanel'
import Overview from './components/Overview'
import RiskPanel from './components/RiskPanel'
import ReportPanel from './components/ReportPanel'
import LoadingOverlay from './components/LoadingOverlay'
import Toast from './components/Toast'

const PANELS = [Portfolio, Backtest, NewsPanel, Overview, RiskPanel, ReportPanel]

export default function App() {
  const { activeTab } = useStore()
  const [toast, setToast] = useState(null)

  const showToast = (msg, type = 'ok') => {
    setToast({ msg, type, key: Date.now() })
    setTimeout(() => setToast(null), 4500)
  }

  useEffect(() => {
    api.latestReport().then(d => {
      if (d.content) useStore.getState().setReport(d.content)
    }).catch(() => {})
  }, [])

  const Panel = PANELS[activeTab]

  return (
    <div className="flex flex-col h-full bg-bg0">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar showToast={showToast} />
        <main className="flex-1 overflow-hidden relative">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.1 }}
              className="absolute inset-0 overflow-y-auto"
              style={{ padding: '20px 22px 64px' }}
            >
              <Panel />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
      <LoadingOverlay />
      <AnimatePresence>
        {toast && <Toast key={toast.key} msg={toast.msg} type={toast.type} />}
      </AnimatePresence>
    </div>
  )
}
