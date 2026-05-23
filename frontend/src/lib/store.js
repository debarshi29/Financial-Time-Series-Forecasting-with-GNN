import { create } from 'zustand'

export const useStore = create((set, get) => ({
  // Pipeline results
  portfolio:   [],
  riskData:    {},
  macro:       {},
  report:      '',
  runDate:     '',
  variant:     'hybrid',
  hasRun:      false,

  // UI
  loading:     false,
  loadingMsg:  '',
  activeTab:   0,
  sortCol:     'final_score',
  sortAsc:     false,

  setResult: (data, date, variant) => set({
    portfolio: data.portfolio || [],
    riskData:  data.risk_data || {},
    macro:     data.macro_context || {},
    report:    data.report_markdown || '',
    runDate:   date,
    variant,
    hasRun:    true,
  }),

  setLoading: (loading, msg = '') => set({ loading, loadingMsg: msg }),
  setTab:     (activeTab) => set({ activeTab }),
  setReport:  (report) => set({ report }),

  setSort: (col) => set(s => ({
    sortAsc: s.sortCol === col ? !s.sortAsc : false,
    sortCol: col,
  })),
}))
