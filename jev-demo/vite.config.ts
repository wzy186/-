import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
// @ts-expect-error 纯 JS 后端，没有类型声明，故意的
import { handleApi } from './server/api.mjs';

/**
 * dev 模式把同一套 API 挂成中间件，不用另起一个进程。
 * 注意：vite 默认只把 VITE_ 前缀的变量给前端，不会进 process.env，
 * 而 API key 必须留在服务端，所以这里手动把它塞进 process.env。
 */
function apiPlugin() {
  return {
    name: 'cs-router-api',
    configureServer(server: any) {
      server.middlewares.use(async (req: any, res: any, next: any) => {
        if (!(await handleApi(req, res))) next();
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  for (const k of ['TYPESAFE_API_KEY', 'OPENROUTER_API_KEY', 'JEV_MODEL', 'JEV_ENDPOINT']) {
    if (env[k]) process.env[k] = env[k];
  }
  return {
    plugins: [react(), apiPlugin()],
    server: { host: '127.0.0.1', port: 5173 },
  };
});
