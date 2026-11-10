/** 原组合 vs 模拟组合对比（FR-03）：现金与总资产、持仓、行业占比、超限统计。
    全部数值来自后端，前端不做金额/占比计算。 */
import type { Analysis, SimError } from "../types";
import TriggerBadge from "./TriggerBadge";

interface Props {
  orig: Analysis;
  sim: Analysis | null;
  simCash: string | null;
  appliedCount: number;
  error: SimError | null;
  title?: string; // 重开方案时展示方案名
}

function Delta({ value }: { value: number }) {
  if (value === 0) return <span className="delta">±0</span>;
  return (
    <span className={`delta ${value > 0 ? "plus" : "minus"}`}>
      {value > 0 ? `+${value}` : value}
    </span>
  );
}

export default function CompareTable({ orig, sim, simCash, appliedCount, error, title }: Props) {
  const simItems = sim ? new Map(sim.items.map((i) => [i.symbol, i])) : null;
  const industries = [...new Set(orig.items.map((i) => i.industry))];
  const origConc = orig.concentration;
  const simConc = sim ? sim.concentration : null;

  return (
    <div className="card">
      <h2>{title ? `方案「${title}」对比` : "原组合 vs 模拟组合"}</h2>
      <div className="compare-note">
        {error ? (
          <span>⚠ {error.message}</span>
        ) : sim ? (
          <span>已生效 {appliedCount} 笔操作，以下为模拟结果（固定价格、零费用，总资产保持不变）</span>
        ) : (
          <span>添加操作后将在此展示对比结果</span>
        )}
      </div>

      <table>
        <thead>
          <tr>
            <th className="align-left">指标</th>
            <th>原组合</th>
            <th>模拟组合</th>
            <th>变化</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="align-left">现金（元）</td>
            <td>{orig.cash}</td>
            <td>{simCash ?? "—"}</td>
            <td>—</td>
          </tr>
          <tr>
            <td className="align-left">总资产（元）</td>
            <td>{orig.total_assets}</td>
            <td>{sim ? sim.total_assets : "—"}</td>
            <td>{sim && sim.total_assets === orig.total_assets ? <span className="delta">不变</span> : "—"}</td>
          </tr>
          {orig.items.map((item) => {
            const simItem = simItems?.get(item.symbol);
            return (
              <tr key={item.symbol}>
                <td className="align-left">
                  {item.name}
                  <span className="symbol">{item.symbol}</span>
                </td>
                <td>
                  {item.quantity} 股 / {item.market_value} 元 / {item.weight_percent}%{" "}
                  <TriggerBadge triggered={item.single_triggered} />
                </td>
                <td>
                  {simItem ? (
                    <>
                      {simItem.quantity} 股 / {simItem.market_value} 元 / {simItem.weight_percent}%{" "}
                      <TriggerBadge triggered={simItem.single_triggered} />
                    </>
                  ) : (
                    "—"
                  )}
                </td>
                <td>{simItem ? <Delta value={simItem.quantity - item.quantity} /> : "—"}</td>
              </tr>
            );
          })}
          {industries.map((industry) => {
            const origRow = orig.items.find((i) => i.industry === industry)!;
            const simRow = simItems
              ? [...simItems.values()].find((i) => i.industry === industry)
              : null;
            const origTrig = origConc.industry.some((c) => c.industry === industry);
            const simTrig = simConc ? simConc.industry.some((c) => c.industry === industry) : false;
            return (
              <tr key={`ind-${industry}`}>
                <td className="align-left">行业：{industry}</td>
                <td>
                  {origRow.industry_weight_percent}% <TriggerBadge triggered={origTrig} />
                </td>
                <td>
                  {simRow ? (
                    <>
                      {simRow.industry_weight_percent}% <TriggerBadge triggered={simTrig} />
                    </>
                  ) : (
                    "—"
                  )}
                </td>
                <td>—</td>
              </tr>
            );
          })}
          <tr>
            <td className="align-left">超限统计（标的 / 行业）</td>
            <td>
              {origConc.single_instrument.length} / {origConc.industry.length}
            </td>
            <td>{simConc ? `${simConc.single_instrument.length} / ${simConc.industry.length}` : "—"}</td>
            <td>—</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
