/**
 * 剧本回归测试：把 12 个内置剧本整段喂给 Mock 引擎，断言路由和拦截结果。
 * 改了关键词表或决策表而没注意到副作用时，这里会先炸。
 */
import { describe, expect, it } from 'vitest';
// @ts-expect-error 纯 JS 后端模块
import { mockJudge } from '../server/mock.mjs';
// @ts-expect-error 纯 JS 后端模块
import { buildState, normalize } from '../server/api.mjs';
import policiesJson from '../shared/policies.json';
import routingJson from '../shared/routing.json';
import scriptsJson from '../shared/scripts.json';
import { checkIntercept, decide } from '../src/lib/decide';
import type { Policies, Routing, ScriptDef } from '../src/types';

const policies = policiesJson as unknown as Policies;
const routing = routingJson as unknown as Routing;
const scripts = (scriptsJson as unknown as { scripts: ScriptDef[] }).scripts;

function run(script: ScriptDef) {
  const hist: { role: string; text: string }[] = [];
  const steps: any[] = [];
  for (const m of script.messages) {
    hist.push(m);
    const group = m.role as 'user' | 'agent';
    const defs = policies.groups[group].questions;
    const state = buildState(group, hist);
    const raw = mockJudge(state, defs, m.text);
    const answers = normalize(raw.answers, defs);
    steps.push({
      role: m.role,
      answers,
      decision: m.role === 'user' ? decide(answers, routing) : null,
      intercept: m.role === 'agent' ? checkIntercept(answers, routing) : null,
    });
  }
  return steps;
}

const byId = (id: string) => scripts.find((s) => s.id === id)!;
const lastUser = (steps: any[]) => [...steps].reverse().find((s) => s.role === 'user');
const lastAgent = (steps: any[]) => [...steps].reverse().find((s) => s.role === 'agent');

describe('剧本端到端', () => {
  it('① 普通查件不该升级', () => {
    const s = run(byId('normal-logistics'));
    expect(s[0].decision.queueId).toBe('bot');
    expect(s[0].decision.human).toBe(false);
    expect(s[1].intercept).toBeNull();
  });

  it('② 情绪一路爬升，最后进客诉专席', () => {
    const s = run(byId('angry-refund'));
    const emo = s.filter((x: any) => x.role === 'user').map((x: any) => x.answers.emotion.value);
    expect(emo[emo.length - 1]).toBeGreaterThan(emo[0]);
    expect(lastUser(s).decision.queueId).toBe('complaint');
    expect(lastUser(s).decision.human).toBe(true);
  });

  it('③ 丢件 + 手术急用 → 高优先专线 R4', () => {
    const s = run(byId('lost-package-urgent'));
    const d = lastUser(s).decision;
    expect(d.ruleId).toBe('R4');
    expect(d.queueId).toBe('vip');
    expect(lastUser(s).answers.urgency.value).toBeGreaterThanOrEqual(2);
  });

  it('④ 12315 + 微博 → R1，P0，必须人工', () => {
    const s = run(byId('regulator-threat'));
    const d = lastUser(s).decision;
    expect(d.ruleId).toBe('R1');
    expect(d.priority).toBe('P0');
    expect(lastUser(s).answers.user_risk.value).toBe('监管投诉');
  });

  it('⑤ 保证三天到 → 绝对化承诺，warn 级', () => {
    const ic = lastAgent(run(byId('agent-overcommit'))).intercept;
    expect(ic.kind).toBe('绝对化承诺');
    expect(ic.level).toBe('warn');
  });

  it('⑥ 私自承诺赔偿 → 越权赔付，block 级', () => {
    const ic = lastAgent(run(byId('agent-overpay'))).intercept;
    expect(ic.kind).toBe('越权赔付');
    expect(ic.level).toBe('block');
  });

  it('⑦ 念出他人手机号地址 → 泄露隐私，block 级', () => {
    const ic = lastAgent(run(byId('agent-leak'))).intercept;
    expect(ic.kind).toBe('泄露隐私');
    expect(ic.level).toBe('block');
  });

  it('⑧ 顶撞用户 → 情绪失控 block，且质量打到最低档', () => {
    const a = lastAgent(run(byId('agent-attitude')));
    expect(a.intercept.kind).toBe('情绪失控');
    expect(a.intercept.level).toBe('block');
    expect(a.answers.agent_quality.value).toBeLessThan(1);
  });

  it('⑨ 薅羊毛特征 → 风控核查，不进赔付流程', () => {
    const s = run(byId('wool-party'));
    expect(lastUser(s).answers.user_risk.value).toBe('疑似薅羊毛');
    expect(lastUser(s).decision.queueId).toBe('riskctrl');
  });

  it('⑩ 优秀范本：质量最高档 + 有共情 + 有方案 + 不拦截', () => {
    const a = lastAgent(run(byId('good-service')));
    expect(a.answers.agent_quality.value).toBeGreaterThan(3);
    expect(a.answers.empathy.value).toBeGreaterThan(0.5);
    expect(a.answers.actionable.value).toBeGreaterThan(0.5);
    expect(a.intercept).toBeNull();
  });

  it('⑪⑫ 无情绪无风险时按意图落到对应组', () => {
    expect(lastUser(run(byId('payment-dup'))).decision.queueId).toBe('payment');
    expect(lastUser(run(byId('presale'))).decision.queueId).toBe('presale');
  });

  it('所有剧本的每条用户消息都能路由，不会返回 null', () => {
    for (const sc of scripts) {
      for (const st of run(sc)) {
        if (st.role === 'user') expect(st.decision, sc.name).not.toBeNull();
      }
    }
  });
});
