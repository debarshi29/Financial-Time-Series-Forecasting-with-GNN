/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        // Flat terminal charcoal — no tint, no gradient
        bg0:  '#0A0A0B',   // page
        bg1:  '#0E0E10',   // panels / header / sidebar
        bg2:  '#151517',   // table head / hover / elevated
        bg3:  '#1C1C1F',   // selected
        // Hairlines
        ln:   '#232327',   // standard hairline
        ln2:  '#2E2E33',   // stronger hairline
        // Text
        tx0:  '#E4E4E7',   // primary
        tx1:  '#A1A1AA',   // secondary
        tx2:  '#71717A',   // muted
        tx3:  '#52525B',   // dim
        tx4:  '#3F3F46',   // dimmest
        // Data colors — used ONLY for meaning
        pos:  '#3DD68C',   // green / BUY / up
        neg:  '#F0616D',   // red / SELL / down
        warn: '#E5A94E',   // amber / caution
        // Single cool accent
        acc:  '#56B6C2',   // cyan — active states, key highlights
      },
      borderRadius: {
        none: '0',
        sm: '2px',
        DEFAULT: '3px',
      },
    },
  },
  plugins: [],
}
