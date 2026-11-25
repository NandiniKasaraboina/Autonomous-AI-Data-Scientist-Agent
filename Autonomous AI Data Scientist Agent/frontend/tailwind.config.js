/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#f5f7f5",
        navy: { DEFAULT: "#ffffff", line: "#e5ebe7", hover: "#f7faf8" },
        forest: { DEFAULT: "#effaf4", line: "#d8eee2", hover: "#e7f7ee" },
        mute: "#748397",
      },
      fontFamily: {
        sans: ["DM Sans", "ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
      },
    },
  },
  plugins: [],
};
