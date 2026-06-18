/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        panel: "#111820",
        panel2: "#17222c",
        scan: "#57d7ff",
        limebeam: "#b9f66a",
        warnline: "#ffb84d",
      },
      boxShadow: {
        scan: "0 0 40px rgba(87, 215, 255, 0.18)",
      },
    },
  },
  plugins: [],
};
