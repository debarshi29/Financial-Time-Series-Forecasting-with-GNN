/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        // Indigo-black palette — distinctive, not generic navy
        s0:  '#07070D',
        s1:  '#0B0B18',
        s2:  '#101022',
        s3:  '#16162C',
        s4:  '#1C1C38',
        bd:  '#242448',
        bds: '#131328',
        // Text
        t0:  '#EAEAFF',
        t1:  '#8080B8',
        t2:  '#505080',
        t3:  '#282850',
        // Accents
        ab:  '#6366F1',   // indigo — primary accent
        ag:  '#22D18E',   // emerald
        ar:  '#F05060',   // rose
        ay:  '#F0A040',   // amber
        ap:  '#A87BFF',   // violet
        ac:  '#22C8E8',   // cyan
      },
    },
  },
  plugins: [],
}
