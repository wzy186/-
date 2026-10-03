import { useState } from 'react';
import type { Msg, Policies } from '../types';
import { AnswerLine, Pill } from './ui';

function emotionTone(v: number): 'ok' | 'warn' | 'bad' {
  return v >= 3 ? 'bad' : v >= 2 ? 'warn' : 'ok';
}

export function MessageBubble({ m, policies }: { m: Msg; policies: Policies | null }) {
  const [open, setOpen] = useState(false);
  const a = m.answers;
  const defs = policies?.groups[m.role === 'user' ? 'user' : 'agent'].questions ?? [];

  const badges: React.ReactNode[] = [];
  if (a?.intent) badges.push(<Pill key="i" tone="accent">{String(a.intent.value)}</Pill>);
  if (a?.emotion)
    badges.push(
      <Pill key="e" tone={emotionTone(a.emotion.value as number)}>
        情绪 {a.emotion.level}
      </Pill>
    );
  if (a?.urgency && (a.urgency.value as number) >= 2)
    badges.push(<Pill key="u" tone="warn">紧急 {a.urgency.level}</Pill>);
  if (a?.user_risk && a.user_risk.value !== '无')
    badges.push(<Pill key="r" tone="bad">风险 {String(a.user_risk.value)}</Pill>);
  if (a?.escalate && (a.escalate.value as number) >= 0.6)
    badges.push(
      <Pill key="es" tone="warn">转人工 {Math.round((a.escalate.value as number) * 100)}%</Pill>
    );
  if (a?.agent_quality)
    badges.push(
      <Pill key="q" tone={(a.agent_quality.value as number) >= 3 ? 'ok' : (a.agent_quality.value as number) >= 2 ? 'default' : 'bad'}>
        质量 {a.agent_quality.level}
      </Pill>
    );
  if (a?.empathy && (a.empathy.value as number) >= 0.5) badges.push(<Pill key="em" tone="ok">有共情</Pill>);
  if (a?.actionable && (a.actionable.value as number) >= 0.5) badges.push(<Pill key="ac" tone="ok">有方案</Pill>);
  if (m.intercept)
    badges.push(
      <Pill key="ic" tone="bad">
        {m.intercept.level === 'block' ? '已拦截' : '风险提示'}·{m.intercept.kind}
      </Pill>
    );

  return (
    <div className={`msg ${m.role}`}>
      <div className="who">{m.role === 'user' ? '用户' : '客服'}</div>
      <div className="bubble-wrap">
        <div className={`bubble ${m.blocked ? 'blocked' : ''}`}>{m.text}</div>

        {m.pending && <div className="tiny muted" style={{ marginTop: 4 }}>判定中…</div>}
        {m.error && <div className="tiny" style={{ color: 'var(--bad)', marginTop: 4 }}>判定失败：{m.error}</div>}

        {m.intercept && (
          <div className={`interceptBox ${m.intercept.level}`}>
            <b>
              {m.intercept.level === 'block' ? '⛔ 已拦截，这句不会发出' : '⚠ 建议修改后再发'}
              ：{m.intercept.kind}（{Math.round(m.intercept.prob * 100)}%）
            </b>
            {m.intercept.rewrite}
          </div>
        )}

        {badges.length > 0 && (
          <div className="badges">
            {badges}
            <button className="pill" onClick={() => setOpen((v) => !v)} style={{ cursor: 'pointer' }}>
              {open ? '收起明细' : '判定明细'}
            </button>
          </div>
        )}

        {open && a && (
          <div className="detail">
            <h4>
              {defs.length} 项判定 · {m.meta?.engine === 'jev' ? `Jev ${m.meta.model}` : 'Mock 规则引擎'}
              {m.meta ? ` · ${m.meta.requests} 次请求 · ${m.meta.ms}ms` : ''}
              {m.meta && m.meta.inputTokens ? ` · ${m.meta.inputTokens} tokens` : ''}
            </h4>
            {defs.map((d) => (a[d.id] ? <AnswerLine key={d.id} label={d.label} a={a[d.id]} /> : null))}
          </div>
        )}
      </div>
    </div>
  );
}
