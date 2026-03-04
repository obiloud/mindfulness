/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
  theme: {
    extend: {
      colors: {
        "mind-bg": "#050816",
        "mind-surface": "#0f172a",
        "mind-accent": "#38bdf8",
      },
    },
  },
  plugins: [],
};

