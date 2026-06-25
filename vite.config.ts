import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    lib: {
      entry: fileURLToPath(new URL("src/index.ts", import.meta.url)),
      name: "ManimJS",
      fileName: (format) => (format === "es" ? "manim-js.js" : "manim-js.umd.cjs"),
      formats: ["es", "umd"],
    },
    cssCodeSplit: false,
    sourcemap: true,
  },
});
