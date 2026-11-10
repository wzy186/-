/** 集中度检查（FR-02）：说明触发对象、实际比例和适用阈值；无超限时显示正常状态。 */
import type { Analysis } from "../types";

interface Props {
  analysis: Analysis;
  thresholds: { single: string; industry: string };
}

export default function ConcentrationPanel({ analysis, thresholds }: Props) {
  const singles = analysis.concentration.single_instrument;
  const industries = analysis.concentration.industry;

  return (
    <div className="card">
      <h2>集中度检查</h2>
      {singles.length === 0 && industries.length === 0 ? (
        <div className="no-alert">
          <span>✓</span>
          <span>
            未发现超限对象（单一标的 &gt; {thresholds.single}%、行业 &gt; {thresholds.industry}% 才触发提醒，等于阈值不触发）
          </span>
        </div>
      ) : (
        <div className="alert-list">
          {singles.map((c) => (
            <div className="alert-item" key={`s-${c.symbol}`}>
              <span className="alert-icon">⚠</span>
              <div className="alert-body">
                <span className="alert-fact">单一标的超限：{c.symbol}</span>
                <span className="alert-nums">
                  实际占比 {c.weight_percent}% &gt; 阈值 {c.threshold_percent}%
                </span>
              </div>
            </div>
          ))}
          {industries.map((c) => (
            <div className="alert-item" key={`i-${c.industry}`}>
              <span className="alert-icon">⚠</span>
              <div className="alert-body">
                <span className="alert-fact">行业超限：{c.industry}</span>
                <span className="alert-nums">
                  实际占比 {c.weight_percent}% &gt; 阈值 {c.threshold_percent}%
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="hint">
        阈值比较使用展示舍入前的精确数值（严格大于才触发）；金额与占比展示保留两位小数。
      </div>
    </div>
  );
}
