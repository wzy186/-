/** 调仓模拟（FR-03）：按顺序添加多笔买卖操作，逐笔展示执行结果与错误。 */
import { useState } from "react";
import type { AppliedOp, Instrument, Op, SimError } from "../types";
import { formatMoney } from "../utils/format";

export type OpStatus = "applied" | "error" | "skipped" | "pending";

interface Props {
  instruments: Instrument[];
  ops: Op[];
  applied: AppliedOp[];
  error: SimError | null;
  onAdd: (op: Op) => void;
  onRemove: (index: number) => void;
  onMove: (index: number, dir: -1 | 1) => void;
  onClear: () => void;
  saveName: string;
  onSaveNameChange: (v: string) => void;
  onSave: () => void;
  saving: boolean;
  saveError: string | null;
  saveOk: string | null;
}

function statusOf(index: number, appliedCount: number, error: SimError | null): OpStatus {
  if (error && error.op_index === index + 1) return "error";
  if (index < appliedCount) return "applied";
  if (error) return "skipped";
  return "pending";
}

export default function SimulationPanel(props: Props) {
  const { instruments, ops, applied, error, onAdd, onRemove, onMove, onClear } = props;
  const [symbol, setSymbol] = useState(instruments[0]?.symbol ?? "");
  const [action, setAction] = useState<"BUY" | "SELL">("BUY");
  const [quantity, setQuantity] = useState("");

  const submit = () => {
    const qty = Number(quantity);
    if (!symbol || !Number.isInteger(qty) || qty <= 0) return;
    onAdd({ action, symbol, quantity: qty });
    setQuantity("");
  };

  return (
    <div className="card">
      <h2>调仓模拟</h2>
      <div className="hint" style={{ marginTop: 0, marginBottom: 12 }}>
        方案从原始组合出发，按顺序逐笔执行；后续操作使用前序结果，校验失败的操作及其后操作不执行。数量必须为正整数。
      </div>

      <div className="op-form">
        <div>
          <label>标的</label>
          <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
            {instruments.map((i) => (
              <option key={i.symbol} value={i.symbol}>
                {i.symbol} {i.name}（{i.industry}，{i.price} 元）
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>方向</label>
          <div className="action-toggle">
            <button
              type="button"
              className={action === "BUY" ? "selected" : ""}
              onClick={() => setAction("BUY")}
            >
              买入
            </button>
            <button
              type="button"
              className={action === "SELL" ? "selected" : ""}
              onClick={() => setAction("SELL")}
            >
              卖出
            </button>
          </div>
        </div>
        <div>
          <label>数量（正整数）</label>
          <input
            type="number"
            min={1}
            step={1}
            value={quantity}
            onChange={(e) => setQuantity(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit();
            }}
            placeholder="数量（正整数）"
          />
        </div>
        <button className="btn btn-primary" onClick={submit}>
          添加操作
        </button>
      </div>

      {ops.length === 0 ? (
        <div className="empty-state">尚未添加操作。选择标的、方向与数量后点击「添加操作」。</div>
      ) : (
        <div className="op-list">
          {ops.map((op, index) => {
            const status = statusOf(index, applied.length, error);
            const detail = index < applied.length ? applied[index] : null;
            const cls = `op-row ${status === "error" ? "error-row" : ""} ${status === "skipped" ? "skipped" : ""}`;
            return (
              <div className={cls} key={index}>
                <span className="op-index">{index + 1}</span>
                <span className={`op-action ${op.action === "BUY" ? "buy" : "sell"}`}>
                  {op.action === "BUY" ? "买入" : "卖出"}
                </span>
                <span>
                  {op.symbol} {instruments.find((i) => i.symbol === op.symbol)?.name ?? ""}
                </span>
                <span className="op-nums">× {op.quantity}</span>
                <span className="op-nums">{detail ? `${formatMoney(detail.amount)} 元` : "—"}</span>
                <span className="op-nums">{detail ? `现金 ${formatMoney(detail.cash_after)}` : "—"}</span>
                <span className="op-controls">
                  {status === "skipped" ? (
                    <span className="op-status-tag">未执行</span>
                  ) : status === "pending" ? (
                    <span className="op-status-tag">待计算</span>
                  ) : (
                    <>
                      <button
                        className="btn btn-sm"
                        disabled={index === 0}
                        onClick={() => onMove(index, -1)}
                        title="上移"
                      >
                        ↑
                      </button>
                      <button
                        className="btn btn-sm"
                        disabled={index === ops.length - 1}
                        onClick={() => onMove(index, 1)}
                        title="下移"
                      >
                        ↓
                      </button>
                      <button className="btn btn-sm" onClick={() => onRemove(index)} title="删除">
                        ✕
                      </button>
                    </>
                  )}
                </span>
                {status === "error" && error && <div className="op-error">⚠ {error.message}</div>}
              </div>
            );
          })}
        </div>
      )}

      <div className="save-box">
        <input
          type="text"
          placeholder="方案名称（必填，1–50 字，不可重复）"
          value={props.saveName}
          onChange={(e) => props.onSaveNameChange(e.target.value)}
          maxLength={50}
        />
        <button
          className="btn btn-primary"
          disabled={ops.length === 0 || Boolean(error) || !props.saveName.trim() || props.saving}
          onClick={props.onSave}
        >
          {props.saving ? "保存中…" : "保存方案"}
        </button>
        {ops.length > 0 && (
          <button className="btn" onClick={onClear}>
            清空操作
          </button>
        )}
      </div>
      {props.saveError && <div className="inline-error">⚠ {props.saveError}</div>}
      {props.saveOk && <div className="inline-ok">✓ {props.saveOk}</div>}
    </div>
  );
}
