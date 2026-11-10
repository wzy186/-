/** 资产结构环形图（SVG）：
    - 分类色跟随行业固定映射，现金用中性灰，颜色不单独表意（配合图例/列表）；
    - 相邻扇区之间保留 2px 表色间隙（stroke 为图表表面色）；
    - 中心显示总资产（千分位）；扇区自带 title 提示。
*/
import { formatMoney } from "../utils/format";

export interface DonutSegment {
  label: string;
  percent: number; // 0~100
  color: string;
  triggered: boolean;
}

interface DonutProps {
  segments: DonutSegment[];
  centerLabel: string;
  centerValue: string; // 已格式化（后端字符串）的金额
}

const SIZE = 140;
const R = 52;
const C = 2 * Math.PI * R;
const GAP_PX = 2.2; // 相邻扇区间隙（≈2px 表色间隙）

export default function Donut({ segments, centerLabel, centerValue }: DonutProps) {
  const total = segments.reduce((sum, s) => sum + s.percent, 0);
  let cum = 0; // 已累计的弧长

  return (
    <div className="donut-wrap">
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="donut" role="img" aria-label={centerLabel}>
        <circle cx={SIZE / 2} cy={SIZE / 2} r={R} fill="none" stroke="var(--seq-100)" strokeWidth={24} />
        {segments
          .filter((s) => s.percent > 0)
          .map((s, i) => {
            const arc = (s.percent / 100) * C;
            const len = Math.max(0, arc - GAP_PX);
            const start = cum + GAP_PX * i; // 每段起点后移一个间隙
            cum += arc;
            return (
              <circle
                key={i}
                cx={SIZE / 2}
                cy={SIZE / 2}
                r={R}
                fill="none"
                stroke={s.color}
                strokeWidth={24}
                strokeDasharray={`${len} ${C - len}`}
                strokeDashoffset={-start}
                transform={`rotate(-90 ${SIZE / 2} ${SIZE / 2})`}
              >
                <title>{`${s.label} ${s.percent.toFixed(2)}%${s.triggered ? "（超限）" : ""}`}</title>
              </circle>
            );
          })}
      </svg>
      <div className="donut-center">
        <div className="donut-center-value">{formatMoney(centerValue)}</div>
        <div className="donut-center-label">{centerLabel}</div>
      </div>
      <span className="sr-note">环图各扇区占比见右侧列表（总计 {total.toFixed(2)}%）</span>
    </div>
  );
}
