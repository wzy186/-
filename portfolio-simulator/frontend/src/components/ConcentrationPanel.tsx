/** 集中度检查（FR-02）：说明触发对象、实际比例和适用阈值；无超限时显示正常状态。 */
import type { Analysis, Portfolio } from "../types";

interface Props {
  analysis: Analysis;
  rules: Portfolio["rules"];
}

export default function ConcentrationPanel({ analysis, rules }: Props) {
  const singles = analysis.concentration.single_instrument;
  const industries = analysis.concentration.industry;

  return (
    <div className="card">
      <h2>集中度检查</h2>
      {singles.length === 0 && industries.length === 0 ? (
        <div className="no-alert">
          <span>✓</span>
          <span>
            未发现超限对象（单一标的 &gt; {rules.single_instrument_threshold}%、行业 &gt;{" "}
            {rules.industry_threshold}% 才触发提醒，等于阈值不触发）
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
        规则：单一标的占比 &gt; {rules.single_instrument_threshold}% 或行业占比 &gt;{" "}
        {rules.industry_threshold}% 触发提醒（比较方式 {rules.comparison}，等于阈值不触发，比较使用展示舍入前的精确值）；
        现金计入总资产分母、不归入任何行业；交易数量为正整数；手续费 {rules.fees} 元。
        金额与占比展示保留两位小数。
      </div>
    </div>
  );
}
