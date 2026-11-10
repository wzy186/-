/** 目标权重调仓建议（扩展功能）：等权分散或自定义目标权重，
    由后端生成整数股数建议并经与人工操作相同的引擎校验，
    展示目标/实际偏差报告，可一键载入操作列表后人工复核保存。 */
import { useEffect, useState } from "react";
import { api, ApiError } from "../api";
import type { Instrument, Op, RebalanceResponse } from "../types";

interface Props {
  instruments: Instrument[];
  currentOpsCount: number;
  onLoadOps: (ops: Op[]) => void;
  autoGenerate: "equal" | "targets" | null; // ?rebalance= URL 参数（验证辅助）
}

export default function RebalancePanel({ instruments, currentOpsCount, onLoadOps, autoGenerate }: Props) {
  const [mode, setMode] = useState<"equal" | "targets">(autoGenerate ?? "equal");
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [result, setResult] = useState<RebalanceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (autoGenerate === "equal") {
      void generate("equal");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const generate = async (m: "equal" | "targets") => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const filled: Record<string, string> = {};
      if (m === "targets") {
        for (const [symbol, value] of Object.entries(inputs)) {
          if (value.trim() !== "") filled[symbol] = value.trim();
        }
      }
      const data = await api.rebalance(m === "equal" ? { mode: "equal" } : { mode: "targets", targets: filled });
      setResult(data);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "建议生成失败，请稍后重试");
    } finally {
      setLoading(false);
    }
  };

  const loadOps = () => {
    if (!result) return;
    if (
      currentOpsCount > 0 &&
      !window.confirm(`将替换当前已添加的 ${currentOpsCount} 笔操作，确定载入建议的 ${result.suggested_ops.length} 笔操作吗？`)
    ) {
      return;
    }
    onLoadOps(result.suggested_ops);
  };

  return (
    <div className="card">
      <h2>目标权重调仓建议</h2>
      <div className="hint" style={{ marginTop: 0, marginBottom: 12 }}>
        输入目标权重，由后端按固定价格计算整数股数的买卖建议（先卖后买，现金不足自动减量），
        建议经与人工操作完全相同的校验；载入后仍可编辑并命名保存。
      </div>

      <div className="op-form" style={{ gridTemplateColumns: "150px 1fr auto" }}>
        <div>
          <label>模式</label>
          <div className="action-toggle">
            <button
              type="button"
              className={mode === "equal" ? "selected" : ""}
              onClick={() => setMode("equal")}
            >
              等权分散
            </button>
            <button
              type="button"
              className={mode === "targets" ? "selected" : ""}
              onClick={() => setMode("targets")}
            >
              自定义权重
            </button>
          </div>
        </div>
        {mode === "targets" ? (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {instruments.map((i) => (
              <div key={i.symbol} style={{ width: 120 }}>
                <label>
                  {i.symbol}（%）<span className="industry-tag">留空=保持</span>
                </label>
                <input
                  type="text"
                  inputMode="decimal"
                  value={inputs[i.symbol] ?? ""}
                  placeholder="如 20"
                  onChange={(e) => setInputs((prev) => ({ ...prev, [i.symbol]: e.target.value }))}
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="industry-tag" style={{ alignSelf: "end", paddingBottom: 8 }}>
            全部标的各占 1/{instruments.length}，现金目标 0%；可一键消除集中度超限
          </div>
        )}
        <div style={{ alignSelf: "end" }}>
          <button className="btn btn-primary" disabled={loading} onClick={() => void generate(mode)}>
            {loading ? "计算中…" : "生成建议"}
          </button>
        </div>
      </div>

      {error && <div className="inline-error">⚠ {error}</div>}

      {result && (
        <div>
          <div className="compare-note">
            <span>
              建议 {result.suggested_ops.length} 笔操作（
              {result.suggested_ops.filter((o) => o.action === "SELL").length} 卖 /{" "}
              {result.suggested_ops.filter((o) => o.action === "BUY").length} 买）
              {result.error ? `，⚠ ${result.error.message}` : "，全部经引擎校验通过"}
            </span>
            <span className="spacer" />
            <button className="btn" onClick={loadOps}>
              载入操作列表
            </button>
          </div>
          <table>
            <thead>
              <tr>
                <th className="align-left">标的</th>
                <th>目标占比</th>
                <th>实际占比</th>
                <th>偏差</th>
              </tr>
            </thead>
            <tbody>
              {result.deviation.map((row) => (
                <tr key={row.symbol}>
                  <td className="align-left">
                    {row.name}
                    <span className="symbol">{row.symbol === "CASH" ? "" : row.symbol}</span>
                  </td>
                  <td>{row.target_percent}%</td>
                  <td>{row.actual_percent}%</td>
                  <td>
                    <span className={`delta ${row.deviation_percent.startsWith("-") ? "minus" : "plus"}`}>
                      {row.deviation_percent}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
