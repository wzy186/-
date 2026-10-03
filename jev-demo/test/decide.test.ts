import { describe, expect, it } from 'vitest';
import { checkIntercept, decide, matchCondition } from '../src/lib/decide';
import routingJson from '../shared/routing.json';
import type { Answers, Routing } from '../src/types';

const routing = routingJson as unknown as Routing;

const A = (o: Record<string, any>): Answers => {
  const out: Answers = {};
  for (const [k, v] of Object.entries(o)) {
    if (typeof v === 'string') out[k] = { type: 'choice', value: v, confidence: 0.9, probs: { [v]: 0.9 } };
    else if (k === 'escalate') out[k] = { type: 'noul', value: v, confidence: 0.8, probs: {} };
    else out[k] = { type: 'score', value: v, level: String(v), levels: [], confidence: 0.8, probs: {} };
  }
  return out;
};

describe('条件匹配', () => {
  it('in 匹配 choice', () => {
    expect(matchCondition(A({ intent: '退款退货' }), { q: 'intent', in: ['退款退货'] })).toBe(true);
    expect(matchCondition(A({ intent: '商品咨询' }), { q: 'intent', in: ['退款退货'] })).toBe(false);
  });
  it('gte 匹配 score 与 noul', () => {
    expect(matchCondition(A({ emotion: 3.2 }), { q: 'emotion', gte: 2 })).toBe(true);
    expect(matchCondition(A({ escalate: 0.7 }), { q: 'escalate', gte: 0.6 })).toBe(true);
    expect(matchCondition(A({ escalate: 0.4 }), { q: 'escalate', gte: 0.6 })).toBe(false);
  });
  it('any 是或的关系', () => {
    const c = { any: [{ q: 'user_risk', in: ['人身攻击'] }, { q: 'emotion', gte: 4 }] } as const;
    expect(matchCondition(A({ user_risk: '无', emotion: 4.1 }), c as any)).toBe(true);
    expect(matchCondition(A({ user_risk: '无', emotion: 1 }), c as any)).toBe(false);
  });
  it('缺失的判定不算命中', () => {
    expect(matchCondition(A({}), { q: 'emotion', gte: 0 })).toBe(false);
  });
});

describe('路由决策', () => {
  it('R1 监管信号压过一切，且必须人工', () => {
    const d = decide(A({ intent: '商品咨询', user_risk: '监管投诉', emotion: 0, urgency: 0, escalate: 0 }), routing)!;
    expect(d.ruleId).toBe('R1');
    expect(d.queueId).toBe('complaint');
    expect(d.human).toBe(true);
    expect(d.priority).toBe('P0');
  });
  it('R2 暴怒即使没有外部信号也进客诉专席', () => {
    const d = decide(A({ intent: '退款退货', user_risk: '无', emotion: 4.2, urgency: 1, escalate: 0.5 }), routing)!;
    expect(d.ruleId).toBe('R2');
    expect(d.queueId).toBe('complaint');
  });
  it('R3 薅羊毛先转风控，排在高情绪规则前面', () => {
    const d = decide(A({ intent: '退款退货', user_risk: '疑似薅羊毛', emotion: 2.5, urgency: 2.5, escalate: 0.9 }), routing)!;
    expect(d.ruleId).toBe('R3');
    expect(d.queueId).toBe('riskctrl');
  });
  it('R4 情绪+紧急双高 → 高优先专线', () => {
    const d = decide(A({ intent: '物流查询', user_risk: '无', emotion: 2.4, urgency: 2.6, escalate: 0.3 }), routing)!;
    expect(d.ruleId).toBe('R4');
    expect(d.queueId).toBe('vip');
  });
  it('R5 模型说该转人工 → 按意图落到对应组', () => {
    const d = decide(A({ intent: '支付问题', user_risk: '无', emotion: 1, urgency: 1, escalate: 0.72 }), routing)!;
    expect(d.ruleId).toBe('R5');
    expect(d.queueId).toBe('payment');
    expect(d.human).toBe(true);
  });
  it('R8 兜底：普通咨询走自助', () => {
    const d = decide(A({ intent: '商品咨询', user_risk: '无', emotion: 0.3, urgency: 0.4, escalate: 0.15 }), routing)!;
    expect(d.ruleId).toBe('R8');
    expect(d.queueId).toBe('presale');
    expect(d.human).toBe(false);
  });
  it('每个意图都能映射到一个存在的队列', () => {
    const ids = new Set(routing.queues.map((q) => q.id));
    for (const [intent, qid] of Object.entries(routing.intentQueue)) {
      expect(ids.has(qid), `${intent} -> ${qid}`).toBe(true);
    }
  });
  it('决策表一定有兜底规则，不会返回 null', () => {
    expect(decide(A({ intent: '闲聊其他', user_risk: '无', emotion: 0, urgency: 0, escalate: 0 }), routing)).not.toBeNull();
  });
});

describe('危险话术拦截', () => {
  const agent = (kind: string, p: number): Answers => ({
    agent_risk: { type: 'choice', value: kind, confidence: p, probs: { [kind]: p } },
  });
  it('无风险放行', () => {
    expect(checkIntercept(agent('无风险', 0.95), routing)).toBeNull();
  });
  it('越权赔付是 block 级', () => {
    const i = checkIntercept(agent('越权赔付', 0.8), routing)!;
    expect(i.level).toBe('block');
    expect(i.rewrite).toContain('售后政策');
  });
  it('绝对化承诺只是 warn', () => {
    expect(checkIntercept(agent('绝对化承诺', 0.7), routing)!.level).toBe('warn');
  });
  it('概率不到阈值就不拦', () => {
    expect(checkIntercept(agent('越权赔付', 0.3), routing)).toBeNull();
  });
  it('每种拦截类型都有改写建议', () => {
    for (const kind of Object.keys(routing.intercept.blockLevels)) {
      expect(routing.intercept.rewrite[kind], kind).toBeTruthy();
    }
  });
});
