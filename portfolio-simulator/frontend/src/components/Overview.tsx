/** 组合概览（FR-01）：总资产/现金/持仓市值/持仓数量统计卡、持仓明细表、行业分布。 */
import type { Portfolio } from "../types";
import TriggerBadge from "./TriggerBadge";
import WeightBar from "./WeightBar";
import { industryColor } from "./industryColors";

export default function Overview({ portfolio }: { portfolio: Portfolio }) {
  const { analysis, instruments, rules } = portfolio;

  const industries = [...new Set(analysis.items.map((item) => item.industry))];

  return (
    <div>
      <div className="stat-tiles">
        <div className="stat-tile hero">
          <div className="label">总资产（元）</div>
          <div className="value">{analysis.total_assets}</div>
          <div className="note">行情快照 {portfolio.snapshot_id}</div>
        </div>
        <div className="stat-tile">
          <div className="label">现金（元）</div>
          <div className="value">{analysis.cash}</div>
          <div className="note">占总资产 {Number(analysis.cash_weight) * 100}%</div>
        </div>
        <div className="stat-tile">
          <div className="label">持仓市值（元）</div>
          <div className="value">{analysis.instruments_value}</div>
          <div className="note">币种 {portfolio.currency}</div>
        </div>
        <div className="stat-tile">
          <div className="label">持仓数量</div>
          <div className="value">{analysis.holdings_count}</div>
          <div className="note">共 {instruments.length} 个给定标的</div>
        </div>
      </div>

      <div className="card">
        <h2>持仓明细</h2>
        <table>
          <thead>
            <tr>
              <th className="align-left">标的</th>
              <th>价格（元）</th>
              <th>数量</th>
              <th>市值（元）</th>
              <th>占比</th>
              <th>单一标的集中度</th>
            </tr>
          </thead>
          <tbody>
            {analysis.items.map((item) => (
              <tr key={item.symbol}>
                <td className="align-left">
                  {item.name}
                  <span className="symbol">{item.symbol}</span>
                  <div className="industry-tag">{item.industry}</div>
                </td>
                <td>{(instruments.find((i) => i.symbol === item.symbol)?.price ?? "—")}</td>
                <td>{item.quantity}</td>
                <td>{item.market_value}</td>
                <td>
                  <WeightBar
                    percent={item.weight_percent}
                    threshold={rules.single_instrument_threshold}
                    triggered={item.single_triggered}
                  />
                </td>
                <td>
                  <TriggerBadge triggered={item.single_triggered} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2>行业分布</h2>
        <div className="industry-list">
          {industries.map((industry) => {
            const item = analysis.items.find((i) => i.industry === industry)!;
            const conc = analysis.concentration.industry.find((c) => c.industry === industry);
            const triggered = Boolean(conc);
            return (
              <div className="industry-row" key={industry}>
                <div className="industry-name">{industry}</div>
                <WeightBar
                  percent={item.industry_weight_percent}
                  threshold={rules.industry_threshold}
                  triggered={triggered}
                  fillColor={industryColor(industry)}
                />
                <div className="industry-value">
                  {item.industry_weight_percent}%
                  {triggered && <span className="delta minus">⚠ 超限</span>}
                </div>
              </div>
            );
          })}
        </div>
        <div className="hint">
          行业占比 = 同行业标的市值之和 ÷ 总资产；现金计入总资产分母，不归入任何行业。
        </div>
      </div>
    </div>
  );
}
