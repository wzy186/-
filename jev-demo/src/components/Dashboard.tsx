import type { Answers, JudgeMeta, Msg, Policies, Routing } from '../types';
import { emotionSeries } from '../lib/decide';
import { mask } from '../lib/mask';
import { AnswerLine, Meter, Pill, Sparkline } from './ui';

interface Props {
  msgs: Msg[];
  routing: Routing | null;
  policies: Policies | null;
  session: { answers: Answers; meta: JudgeMeta } | null;
  onAnalyze: () => void;
  analyzing: boolean;
  usage: { requests: number; inputTokens: number; costUsd: number; ms: number };
}

export function Dashboard({ msgs, routing, policies, session, onAnalyze, analyzing, usage }: Props) {
  const lastUser = [...msgs].reverse().find((m) => m.role === 'user' && m.answers);
  const decision = lastUser?.decision || null;
  const series = emotionSeries(msgs);
  const peak = series.length ? Math.max(...series.map((p) => p.v)) : 0;

  const risks = msgs.filter(
    (m) =>
      (m.answers?.user_risk && m.answers.user_risk.value !== '无') ||
      m.intercept
  );

  const agentMsgs = msgs.filter((m) => m.role === 'agent' && m.answers?.agent_quality);
  const avgQuality = agentMsgs.length
    ? agentMsgs.reduce((a, m) => a + (m.answers!.agent_quality.value as number), 0) / agentMsgs.length
    : null;
  const empathyRate = agentMsgs.length
    ? agentMsgs.filter((m) => (m.answers!.empathy?.value as number) >= 0.5).length / agentMsgs.length
    : null;

  const sessionDefs = policies?.groups.session.questions ?? [];

  return (
    <>
      <div className="card">
        <h2>当前路由</h2>
        <p className="hint">按最近一条用户消息的判定结果走决策表</p>
        {decision ? (
          <div className={`routeCard ${decision.priority}`}>
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <span className="q">{decision.queueName}</span>
              <span className="row" style={{ gap: 4 }}>
                <Pill tone={decision.priority === 'P0' ? 'bad' : decision.priority === 'P1' ? 'warn' : 'default'}>
                  {decision.priority}
                </Pill>
                <Pill tone={decision.human ? 'warn' : 'ok'}>{decision.human ? '需人工' : '可自助'}</Pill>
              </span>
            </div>
            <div className="tiny muted" style={{ marginTop: 2 }}>
              响应目标 {decision.sla} · 命中 {decision.ruleId} {decision.ruleName}
            </div>
            <div className="why">{decision.reason}</div>
          </div>
        ) : (
          <div className="empty">还没有用户消息。发一条，或者从左边点一个剧本。</div>
        )}
      </div>

      <div className="card">
        <h2>用户情绪趋势</h2>
        <p className="hint">0 平静 → 4 暴怒，虚线是「明显不满」的告警线</p>
        <Sparkline points={series} />
        <div className="kpis" style={{ marginTop: 8 }}>
          <div className="kpi">
            <b>{series.length ? series[series.length - 1].v.toFixed(1) : '—'}</b>
            <span>当前情绪</span>
          </div>
          <div className="kpi">
            <b style={{ color: peak >= 3 ? 'var(--bad)' : peak >= 2 ? 'var(--warn)' : undefined }}>
              {series.length ? peak.toFixed(1) : '—'}
            </b>
            <span>峰值</span>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>风险与拦截 {risks.length > 0 && <Pill tone="bad">{risks.length}</Pill>}</h2>
        <p className="hint">用户侧升级信号 + 客服侧危险话术</p>
        {risks.length === 0 ? (
          <div className="empty">暂无风险信号。</div>
        ) : (
          <div className="feed">
            {risks.map((m) => {
              const ic = m.intercept;
              const ur = m.answers?.user_risk;
              return (
                <div className={`feedItem ${ic?.level === 'warn' ? 'warn' : ''}`} key={m.id}>
                  <b>
                    {ic
                      ? `客服·${ic.kind}（${ic.level === 'block' ? '已拦截' : '提示'}）`
                      : `用户·${String(ur?.value)}`}
                  </b>
                  <span>{mask(m.text).slice(0, 46)}{m.text.length > 46 ? '…' : ''}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="card">
        <h2>客服服务质量</h2>
        <p className="hint">逐句打分的滚动均值，落到质检抽样用</p>
        {avgQuality == null ? (
          <div className="empty">还没有客服消息。切到「客服」身份发一条试试。</div>
        ) : (
          <>
            <div className="row" style={{ justifyContent: 'space-between', marginBottom: 4 }}>
              <span className="muted tiny">均分</span>
              <b>{avgQuality.toFixed(2)} / 4</b>
            </div>
            <Meter
              value={avgQuality}
              max={4}
              tone={avgQuality >= 3 ? 'var(--ok)' : avgQuality >= 2 ? 'var(--warn)' : 'var(--bad)'}
              left="很差"
              right="优秀"
            />
            <div className="kpis" style={{ marginTop: 8 }}>
              <div className="kpi">
                <b>{Math.round((empathyRate ?? 0) * 100)}%</b>
                <span>共情覆盖率</span>
              </div>
              <div className="kpi">
                <b>{agentMsgs.length}</b>
                <span>已质检条数</span>
              </div>
            </div>
          </>
        )}
      </div>

      <div className="card">
        <h2>会话级分析</h2>
        <p className="hint">拿整段对话再跑一次，用于归档、CSAT 预测和质检抽样</p>
        <button className="btn ghost tiny" onClick={onAnalyze} disabled={analyzing || msgs.length === 0}>
          {analyzing ? '分析中…' : '分析整段会话'}
        </button>
        {session && (
          <div className="detail" style={{ marginTop: 8 }}>
            {sessionDefs.map((d) =>
              session.answers[d.id] ? <AnswerLine key={d.id} label={d.label} a={session.answers[d.id]} /> : null
            )}
          </div>
        )}
      </div>

      <div className="card">
        <h2>本次会话开销</h2>
        <div className="usage">
          请求 {usage.requests} 次 · input {usage.inputTokens} tokens · 累计 {usage.ms}ms
          <br />
          估算费用 ${usage.costUsd.toFixed(6)}（$0.042 / 1M input tokens，output 暂不计费）
        </div>
        {routing && (
          <details className="raw" style={{ marginTop: 8 }}>
            <summary>当前决策表（{routing.rules.length} 条规则 / {routing.queues.length} 个队列）</summary>
            <pre>
              {routing.rules.map((r) => `${r.id}  ${r.name}\n      → ${r.then.queue} ${r.then.priority}`).join('\n')}
            </pre>
          </details>
        )}
      </div>
    </>
  );
}
