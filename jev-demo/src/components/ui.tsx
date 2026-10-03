import type { Answer } from '../types';

export function Pill({
  tone = 'default',
  children,
  title,
}: {
  tone?: 'default' | 'ok' | 'warn' | 'bad' | 'accent';
  children: React.ReactNode;
  title?: string;
}) {
  return (
    <span className={`pill ${tone === 'default' ? '' : tone}`} title={title}>
      {children}
    </span>
  );
}

export function ProbBars({ probs, top = 4 }: { probs: Record<string, number>; top?: number }) {
  const rows = Object.entries(probs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, top);
  if (!rows.length) return null;
  return (
    <div className="probs">
      {rows.map(([k, v], i) => (
        <div className="prob" key={k}>
          <span title={k} style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {k}
          </span>
          <span className={`t ${i === 0 ? '' : 'dim'}`}>
            <i style={{ width: `${Math.round(v * 100)}%` }} />
          </span>
          <span className="n">{Math.round(v * 100)}%</span>
        </div>
      ))}
    </div>
  );
}

/** 一条判定的展开明细 */
export function AnswerLine({ label, a }: { label: string; a: Answer }) {
  let headline = '';
  if (a.type === 'choice') headline = String(a.value);
  else if (a.type === 'noul') headline = `${Math.round((a.value as number) * 100)}%`;
  else headline = `${a.level}（${(a.value as number).toFixed(2)}）`;
  return (
    <div className="qline">
      <div>
        <span className="muted">{label}</span>
        <b>{headline}</b>
      </div>
      <ProbBars probs={a.probs} />
      {a.confidence != null && (
        <div className="tiny muted" style={{ marginTop: 2 }}>
          置信度 {Math.round(a.confidence * 100)}%
        </div>
      )}
    </div>
  );
}

/** 情绪 / 质量这类档位分值的条 */
export function Meter({
  value,
  max,
  tone,
  left,
  right,
}: {
  value: number;
  max: number;
  tone: string;
  left: string;
  right: string;
}) {
  return (
    <>
      <div className="meter">
        <i style={{ width: `${Math.min(100, (value / max) * 100)}%`, background: tone }} />
      </div>
      <div className="legend">
        <span>{left}</span>
        <span>{right}</span>
      </div>
    </>
  );
}

/** 情绪折线，纯内联 SVG，不引图表库 */
export function Sparkline({ points, max = 4 }: { points: { i: number; v: number }[]; max?: number }) {
  if (points.length < 2) return <div className="empty">至少两条用户消息才画得出趋势</div>;
  const w = 320;
  const h = 56;
  const pad = 4;
  const dx = (w - pad * 2) / (points.length - 1);
  const y = (v: number) => h - pad - (v / max) * (h - pad * 2);
  const d = points.map((p, i) => `${i ? 'L' : 'M'}${pad + i * dx},${y(p.v)}`).join(' ');
  const last = points[points.length - 1];
  const tone = last.v >= 3 ? 'var(--bad)' : last.v >= 2 ? 'var(--warn)' : 'var(--ok)';
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} role="img" aria-label="用户情绪趋势">
      <line x1={pad} y1={y(2)} x2={w - pad} y2={y(2)} stroke="var(--line)" strokeDasharray="3 3" />
      <path d={d} fill="none" stroke={tone} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
      {points.map((p, i) => (
        <circle key={i} cx={pad + i * dx} cy={y(p.v)} r={i === points.length - 1 ? 3.5 : 2.2} fill={tone} />
      ))}
    </svg>
  );
}
