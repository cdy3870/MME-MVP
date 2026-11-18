import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const examples = fileURLToPath(new URL("../schema/examples/report/valid", import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@examples": examples } },
  server: {
    fs: { allow: [".."] }, // the example reports live outside this folder
    proxy: { "/api": "http://127.0.0.1:8000" }, // `migration-eval serve` listens here: one origin, no CORS
  },
});
