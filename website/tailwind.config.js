/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Onest', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        bg: '#F9F9F7',
        surface: '#FFFFFF',
        border: '#E5E5E3',
        ink: '#111110',
        muted: '#6F6F6B',
        'status-green': '#16A34A',
        'status-red': '#DC2626',
        'status-yellow': '#CA8A04',
        'status-blue': '#2563EB',
      },
    },
  },
  plugins: [],
};
