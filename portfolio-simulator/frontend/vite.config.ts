import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 开发模式：/api 代理到后端 8000 端口；生产模式由 FastAPI 直接托管 dist/。
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
  },
});
