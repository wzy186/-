/**
 * 生产模式服务：serve dist/ + /api/*。
 *
 *   npm run build && npm start
 *
 * dev 模式不需要它，vite.config.ts 里已经把同一套 API 挂成中间件了。
 */
import { createServer } from 'node:http';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { extname, join, normalize as npath, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { handleApi } from './api.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const DIST = join(HERE, '..', 'dist');
const PORT = Number(process.env.PORT || 5180);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
};

createServer(async (req, res) => {
  if (await handleApi(req, res)) return;

  if (!existsSync(DIST)) {
    res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
    return res.end('还没构建。先跑 npm run build，或者用 npm run dev。');
  }
  const url = new URL(req.url, 'http://localhost');
  let file = join(DIST, npath(decodeURIComponent(url.pathname)));
  if (!file.startsWith(DIST)) file = DIST; // 目录穿越防护
  if (!existsSync(file) || statSync(file).isDirectory()) file = join(DIST, 'index.html');
  res.writeHead(200, { 'Content-Type': MIME[extname(file)] || 'application/octet-stream' });
  createReadStream(file).pipe(res);
}).listen(PORT, '127.0.0.1', () => {
  const hasKey = Boolean(process.env.OPENROUTER_API_KEY || process.env.TYPESAFE_API_KEY);
  const engine = hasKey ? 'Jev 真实接口' : 'Mock（未配置 API_KEY）';
  console.log(`客服意图路由沙盘  http://127.0.0.1:${PORT}   判定引擎：${engine}`);
});
