import { defineConfig } from "vite";

// dev: `bun run dev` alongside `uv run tgsearch.py serve`; prod: `bun run build`, then `serve` hosts dist/
const api = "http://127.0.0.1:8000";
export default defineConfig({
  server: { proxy: { "/api": api, "/media": api, "/queries": api } },
  build: {
    rolldownOptions: {
      // Mantine marks every module "use client" (a React Server Components hint, meaningless in an SPA)
      onwarn: (w, warn) => w.code !== "MODULE_LEVEL_DIRECTIVE" && warn(w),
    },
  },
});
