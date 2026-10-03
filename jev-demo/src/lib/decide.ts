import type { Answers, Condition, Decision, Intercept, Routing } from '../types';

/**
 * 路由决策引擎：纯函数，输入判定结果 + 决策表，输出去哪个队列。
 * 规则从上往下匹配，第一条命中即生效。这样风险规则放前面就自然压过业务规则。
 */

/** 取某个判定的可比较值：choice→标签，score→档位下标，noul→概率 */
export function valueOf(answers: Answers, qid: string): string | number | null {
  const a = answers[qid];
  if (!a) return null;
  if (a.type === 'choice') return a.value as string;
  if (a.type === 'noul') return a.value as number;
  return a.value as number; // score
}

export function matchCondition(answers: Answers, c: Condition): boolean {
  if ('any' in c) return c.any.some((x) => matchCondition(answers, x));
  const v = valueOf(answers, c.q);
  if (v === null) return false;
  if ('in' in c) return typeof v === 'string' && c.in.includes(v);
  if ('gte' in c) return typeof v === 'number' && v >= c.gte;
  return false;
}

export function decide(answers: Answers, routing: Routing): Decision | null {
  for (const rule of routing.rules) {
    if (!rule.when.every((c) => matchCondition(answers, c))) continue;
    let queueId = rule.then.queue;
    if (queueId === 'byIntent') {
      const intent = valueOf(answers, 'intent');
      queueId = (typeof intent === 'string' && routing.intentQueue[intent]) || 'bot';
    }
    const q = routing.queues.find((x) => x.id === queueId);
    return {
      ruleId: rule.id,
      ruleName: rule.name,
      queueId,
      queueName: q?.name || queueId,
      sla: q?.sla || '—',
      human: rule.then.human,
      priority: rule.then.priority,
      reason: rule.then.reason,
    };
  }
  return null;
}

/** 客服危险话术拦截：返回 null 表示放行 */
export function checkIntercept(answers: Answers, routing: Routing): Intercept | null {
  const a = answers['agent_risk'];
  if (!a || a.type !== 'choice') return null;
  const kind = a.value as string;
  if (kind === '无风险') return null;
  const prob = a.probs[kind] ?? a.confidence ?? 0;
  if (prob < routing.intercept.minProb) return null;
  const level = routing.intercept.blockLevels[kind];
  if (!level) return null;
  return {
    kind,
    level,
    prob,
    rewrite: routing.intercept.rewrite[kind] || '请改用不含承诺与情绪的表述。',
  };
}

/** 情绪趋势：给看板画折线用 */
export function emotionSeries(
  msgs: { role: string; answers?: Answers }[]
): { i: number; v: number }[] {
  const out: { i: number; v: number }[] = [];
  let i = 0;
  for (const m of msgs) {
    if (m.role !== 'user') continue;
    const v = m.answers?.emotion?.value;
    if (typeof v === 'number') out.push({ i: i++, v });
  }
  return out;
}
