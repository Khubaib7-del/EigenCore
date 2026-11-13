import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Relative base so the built files load from file:// inside pywebview later.
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "dist", sourcemap: false, target: "es2022" },
});
