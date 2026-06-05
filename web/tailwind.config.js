/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        base: {
          950: '#080812',
          900: '#0f1117',
          800: '#131825',
          700: '#1a2035',
          600: '#1e2a40',
        },
        border: '#1e2a40',
        accent: {
          DEFAULT: '#6366f1',
          hover:   '#818cf8',
          dim:     '#6366f120',
        },
        cyan: { DEFAULT: '#22d3ee', dim: '#22d3ee20' },
        success: '#22c55e',
        warning: '#f59e0b',
        danger:  '#ef4444',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
