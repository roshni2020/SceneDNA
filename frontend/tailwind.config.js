/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          dark: "#0a0a0f",
          card: "#12131c",
          border: "#1f2233",
          accent: "#6366f1",
          danger: "#f43f5e",
          warning: "#f59e0b",
          success: "#10b981",
        },
      },
      keyframes: {
        'drift': {
          '0%': { transform: 'translate3d(-2%, -1%, 0) scale(1.02)' },
          '50%': { transform: 'translate3d(2%, 1%, 0) scale(1.05)' },
          '100%': { transform: 'translate3d(-2%, -1%, 0) scale(1.02)' },
        },
        'pulse-risk': {
          '0%, 100%': { opacity: '1', boxShadow: '0 0 15px rgba(244, 63, 94, 0.6)' },
          '50%': { opacity: '0.5', boxShadow: '0 0 5px rgba(244, 63, 94, 0.2)' },
        },
      },
      animation: {
        'drift': 'drift 9s ease-in-out infinite',
        'pulse-risk': 'pulse-risk 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
};
