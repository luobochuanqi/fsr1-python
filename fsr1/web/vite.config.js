import { defineConfig } from "vite";

export default defineConfig({
  // 本地开发时 /api 转发给 WSGI 服务（uv run python serve.py）。
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
