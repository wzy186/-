/**
 * API 处理层：dev 用 vite 中间件挂它，prod 用 server/index.mjs 挂它。
 *
 *   GET  /api/config     当前引擎（jev / mock）
 *   GET  /api/policies   判定策略
 *   GET  /api/routing    路由决策表
 *   GET  /api/scripts    内置剧本
 *   POST /api/judge      { group, messages } -> { answers, meta }
 *
 * 前端永远拿到同一种归一化后的结构，不用关心背后是真模型还是 mock。
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { judge as jevJudge } from './jev.mjs';
import { mockJudge, labelOf } from './mock.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const SHARED = join(HERE, '..', 'shared');

const readShared = (f) => JSON.parse(readFileSync(join(SHARED, f), 'utf8'));

// 每次请求都重读，改 json 不用重启
const policies = () => readShared('policies.json');
const routing = () => readShared('routing.json');
const scripts = () => readShared('scripts.json');

const apiKey = () => (process.env.OPENROUTER_API_KEY || process.env.TYPESAFE_API_KEY || '').trim();
const engine = () => (apiKey() ? 'jev' : 'mock');

const ROLE_CN = { user: '用户', agent: '客服' };

/** 拼给模型看的 state：前文 + 明确指出要判定哪一条 */
export function buildState(group, messages, contextTurns = 8) {
  if (group === 'session') {
    return (
      '以下是一通电商客服会话的完整记录。\n\n' +
      messages.map((m) => `${ROLE_CN[m.role] || m.role}：${m.text}`).join('\n')
    );
  }
  const idx = messages.length - 1;
  const ctx = messages.slice(Math.max(0, idx - contextTurns), idx);
  const target = messages[idx];
  const head = ctx.length
    ? '会话上文：\n' + ctx.map((m) => `${ROLE_CN[m.role] || m.role}：${m.text}`).join('\n') + '\n\n'
    : '这是会话的第一条消息。\n\n';
  return `${head}需要判定的这条消息（${ROLE_CN[target.role]}说的）：\n${target.text}`;
}

/** 把 Jev / mock 的原始答案统一成前端好用的形状 */
export function normalize(answers, defs) {
  const out = {};
  for (const q of defs) {
    const a = answers?.[q.id];
    if (!a) continue;
    if (q.type === 'choice') {
      out[q.id] = {
        type: 'choice',
        value: a.choice,
        confidence: a.confidence ?? null,
        probs: a.probabilities || {},
      };
    } else if (q.type === 'noul') {
      out[q.id] = {
        type: 'noul',
        value: typeof a.noul === 'number' ? a.noul : null,
        confidence: a.confidence ?? null,
        probs: {},
      };
    } else if (q.type === 'score') {
      const levels = q.criteria.map(labelOf);
      // Jev 的 probabilities 可能按下标也可能按原文本作 key，都归一到短标签
      const probs = {};
      for (const [k, v] of Object.entries(a.probabilities || {})) {
        const i = Number(k);
        const key = Number.isInteger(i) && levels[i] !== undefined ? levels[i] : labelOf(k);
        probs[key] = (probs[key] || 0) + Number(v);
      }
      const score = typeof a.score === 'number' ? a.score : 0;
      out[q.id] = {
        type: 'score',
        value: score,
        level: levels[Math.max(0, Math.min(levels.length - 1, Math.round(score)))],
        levels,
        confidence: a.confidence ?? null,
        probs,
      };
    }
  }
  return out;
}

async function handleJudge(body) {
  const pol = policies();
  const group = body.group || 'user';
  const defs = pol.groups[group]?.questions;
  if (!defs) throw new Error(`未知判定组：${group}`);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  if (!messages.length) throw new Error('messages 为空');

  const state = buildState(group, messages);
  const focus = group === 'session' ? '' : messages[messages.length - 1].text;

  let raw;
  if (engine() === 'jev' && !body.forceMock) {
    raw = await jevJudge(apiKey(), state, defs, pol.maxQuestionsPerRequest || 5);
  } else {
    raw = mockJudge(state, defs, focus);
  }
  return {
    group,
    answers: normalize(raw.answers, defs),
    raw: body.includeRaw ? raw.answers : undefined,
    state: body.includeState ? state : undefined,
    meta: raw.meta,
  };
}

/** 返回 true 表示这个请求已经被处理掉了 */
export async function handleApi(req, res) {
  const url = new URL(req.url, 'http://localhost');
  if (!url.pathname.startsWith('/api/')) return false;

  const send = (code, obj) => {
    const b = Buffer.from(JSON.stringify(obj), 'utf8');
    res.writeHead(code, {
      'Content-Type': 'application/json; charset=utf-8',
      'Content-Length': b.length,
      'Cache-Control': 'no-store',
    });
    res.end(b);
  };

  try {
    if (req.method === 'GET') {
      switch (url.pathname) {
        case '/api/config':
          return send(200, {
            engine: engine(),
            model: process.env.JEV_MODEL || 'jev-latest',
            hasKey: Boolean(apiKey()),
            maxQuestionsPerRequest: policies().maxQuestionsPerRequest,
          }), true;
        case '/api/policies':
          return send(200, policies()), true;
        case '/api/routing':
          return send(200, routing()), true;
        case '/api/scripts':
          return send(200, scripts()), true;
      }
    }
    if (req.method === 'POST' && url.pathname === '/api/judge') {
      const chunks = [];
      for await (const c of req) chunks.push(c);
      const body = JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}');
      const out = await handleJudge(body);
      return send(200, out), true;
    }
    send(404, { error: 'not found' });
    return true;
  } catch (e) {
    send(500, { error: String(e?.message || e) });
    return true;
  }
}
