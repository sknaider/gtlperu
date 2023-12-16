/** @type {import('tailwindcss').Config} */
const defaultTheme = require("tailwindcss/defaultTheme");
module.exports = {
  darkMode: 'class',
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  theme: {
    extend: {
      colors: {
        theme: {
          light: "#f2f2e6",
          dark: "#191919"
        },
        neoMint: {
          50: "#e5f8e3",
          100: "#defede",
          200: "#cdffd6",
          300: "#bfffd2",
          400: "#b6ffcf",
          500: "#aef8c6",
          600: "#9fdcac",
          700: "#8bb58d",
          800: "#74856f",
          900: "#4d514a",
        },
        nileStone: {
          50: "#d5f2dd",
          100: "#c4f1d7",
          200: "#a3efd6",
          300: "#87e9da",
          400: "#72ded7",
          500: "#64ccc5",
          600: "#5db2a3",
          700: "#58927b",
          800: "#4d6c57",
          900: "#3a443a",
        },
        carbonFiber: {
          50: "#dad5d1",
          100: "#c2bebb",
          200: "#949291",
          300: "#6b6a69",
          400: "#484747",
          500: "#2d2d2d",
          600: "#1c1c1c",
          700: "#141414",
          800: "#121212",
          900: "#141414",
        },
        silverMedal: {
          50: "#f3eee9",
          100: "#f4efeb",
          200: "#f4f0ee",
          300: "#f1eeed",
          400: "#e7e6e6",
          500: "#d6d6d6",
          600: "#bcbcbc",
          700: "#9a9a9a",
          800: "#727272",
          900: "#474747",
        },
        primary: "var(--color-primary)",
        secondary: "var(--color-secondary)",
      },
      textColor: {
        default: "var(--color-text)",
        offset: "var(--color-text-offset)",
        overPrimary: "var(--color-text-over-primary)",
        overSecondary: "var(--color-text-over-secondary)",
      },
      backgroundColor: {
        default: "var(--color-background)",
        offset: "var(--color-background-offset)",
      },
      borderColor: {
        default: "var(--color-border)",
      },
      fontFamily: {
        sans: ["Inter Variable", "Inter", ...defaultTheme.fontFamily.sans],
        plaster: ["Plaster", ...defaultTheme.fontFamily.sans],
        paytone: ["Paytone One", ...defaultTheme.fontFamily.sans],
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
      },
      animation: {
        marquee: "marquee 50s linear infinite",
        morph: "morph 5s ease-in-out infinite",
      },
      keyframes: {
        marquee: {
          from: {
            transform: "translateX(0)",
          },
          to: {
            transform: "translateX(calc(-100% - 2.5rem))",
          },
        },
        morph: {
          from: {
            opacity: "1",
            display: "block"
          },
          to: {
            opacity: "0",
            display: "hidden"
          },
        }
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
