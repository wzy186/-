/** 占比条：轨道为同色阶浅色步（seq-100），填充 4px 圆角数据端、基线端直角；
    阈值刻度为 2px 轴色细线；触发时填充切换为状态色并有图标+文字徽标。 */

interface WeightBarProps {
  percent: string; // "40.00"
  threshold?: string; // "30.00"，展示阈值刻度
  triggered: boolean;
  fillColor?: string; // 分类色（行业分布用）
}

export default function WeightBar({ percent, threshold, triggered, fillColor }: WeightBarProps) {
  const width = Math.max(0, Math.min(100, Number(percent) || 0));
  const tick = threshold !== undefined ? Math.max(0, Math.min(100, Number(threshold) || 0)) : null;
  return (
    <div className="weight-cell">
      <div className="weight-bar">
        <div
          className={`fill ${triggered ? "alert" : fillColor ? "categorical" : ""}`}
          style={{ width: `${width}%`, ...(fillColor && !triggered ? ({ "--fill-color": fillColor } as React.CSSProperties) : {}) }}
        />
        {tick !== null && <div className="tick" style={{ left: `${tick}%` }} title={`阈值 ${threshold}%`} />}
      </div>
      <div className="weight-text">
        <span>{percent}%</span>
        {threshold !== undefined && (
          <span className="industry-tag">阈值 {threshold}%</span>
        )}
      </div>
    </div>
  );
}
