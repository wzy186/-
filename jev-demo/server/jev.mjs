/**
 * Jev（TypeSafe AI System One）调用层。
 *
 *   POST https://api.typesafe.ai/v1/systemone
 *   Authorization: Bearer $TYPESAFE_API_KEY
 *
 * 做了三件事：
 *   1. 按 maxQuestionsPerRequest 把问题切批，多批并发，答案合并
 *   2. 429 / 529 指数退避重试
 *   3. 统计 token 用量与耗时，回给前端展示
 */

export function getEndpoint(apiKey = '') {
  if (process.env.JEV_ENDPOINT) return process.env.JEV_ENDPOINT;
  const key = (apiKey || process.env.OPENROUTER_API_KEY || process.env.TYPESAFE_API_KEY || '').trim();
  if (key.startsWith('sk-or-v1')) {
    return 'https://openrouter.ai/api/alpha/decisions';
  }
  return 'https://api.typesafe.ai/v1/systemone';
}

export function getModel() {
  return process.env.JEV_MODEL || 'jev-latest';
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** 把策略里的问题定义转成 Jev 的 questions 对象 */
export function toJevQuestions(defs) {
  const out = {};
  for (const q of defs) {
    const node = { type: q.type, instructions: q.instructions };
    if (q.type === 'choice') node.criteria = q.criteria;
    else if (q.type === 'score') node.criteria = q.criteria;
    out[q.id] = node;
  }
  return out;
}

function chunk(arr, size) {
  const out = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

async function callOnce(apiKey, state, questions, tries = 5) {
  const endpoint = getEndpoint(apiKey);
  const model = getModel();
  let lastErr = null;
  for (let attempt = 1; attempt <= tries; attempt++) {
    let res;
    try {
      res = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
          'HTTP-Referer': 'https://github.com/cs-intent-router',
          'X-Title': 'CS Intent Router',
        },
        body: JSON.stringify({ model, state, questions }),
        signal: AbortSignal.timeout(60_000),
      });
    } catch (e) {
      lastErr = e;
      await sleep(Math.min(2 ** attempt * 250, 8000));
      continue;
    }
    if (res.status === 429 || res.status === 529) {
      lastErr = new Error(`Jev ${res.status} 限流/过载`);
      await sleep(Math.min(2 ** attempt * 400, 10_000));
      continue;
    }
    if (res.status === 401 || res.status === 403) {
      const isOR = endpoint.includes('openrouter.ai');
      throw new Error(`Jev 鉴权失败（401/403）。请求端点: ${endpoint}，请检查 ${isOR ? 'OPENROUTER_API_KEY / TYPESAFE_API_KEY 是否有效' : 'TYPESAFE_API_KEY'}`);
    }
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`Jev ${res.status}: ${body.slice(0, 300)}`);
    }
    return res.json();
  }
  throw lastErr || new Error('Jev 请求失败');
}

/**
 * @param {string} apiKey
 * @param {string} state      对话上下文
 * @param {Array}  defs       策略里的问题定义数组
 * @param {number} maxPerReq  单次请求最多几个问题
 */
export async function judge(apiKey, state, defs, maxPerReq = 5) {
  const t0 = Date.now();
  const batches = chunk(defs, Math.max(1, maxPerReq));
  const results = await Promise.all(
    batches.map((b) => callOnce(apiKey, state, toJevQuestions(b)))
  );
  const answers = {};
  let inputTokens = 0;
  let outputTokens = 0;
  let costUsd = 0;
  let model = getModel();
  for (const r of results) {
    Object.assign(answers, r.answers || {});
    inputTokens += r.usage?.input_tokens || 0;
    outputTokens += r.usage?.output_tokens || 0;
    if (typeof r.usage?.cost === 'number') {
      costUsd += r.usage.cost;
    }
    if (r.model) model = r.model;
  }
  if (costUsd === 0 && inputTokens > 0) {
    // TypeSafe Direct 官方价：$0.042 / 1M input tokens，output 暂不计费
    costUsd = (inputTokens / 1_000_000) * 0.042;
  }
  return {
    answers,
    meta: {
      engine: 'jev',
      model,
      requests: batches.length,
      inputTokens,
      outputTokens,
      costUsd,
      ms: Date.now() - t0,
    },
  };
}
