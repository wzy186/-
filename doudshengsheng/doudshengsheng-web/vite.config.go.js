import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Go 版前端配置:代理到 8082(Go 后端),端口 5174(和 Java 版 5173 隔离)
// 用法:npm run dev:go
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5174,
    proxy: {
      '/api': {
        target: 'http://localhost:8082',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
