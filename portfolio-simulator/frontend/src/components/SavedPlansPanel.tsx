/** 方案持久化（FR-04）：已保存方案列表、重开查看（只读）、删除与派生草稿。
    原方案为只读快照：载入模拟器后为新的操作草稿，保存时另起新名，不覆盖原方案。 */
import { useState } from "react";
import type { Op, PlanDetail, PlanSummary } from "../types";
import CompareTable from "./CompareTable";

interface Props {
  plans: PlanSummary[];
  opened: PlanDetail | null;
  openError: string | null;
  onOpen: (id: number) => void;
  onClose: () => void;
  onDelete: (id: number) => Promise<string | null>;
  onLoadOps: (ops: Op[]) => void;
}

export default function SavedPlansPanel({ plans, opened, openError, onOpen, onClose, onDelete, onLoadOps }: Props) {
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const handleDelete = async (id: number, name: string) => {
    if (!window.confirm(`确定删除方案「${name}」吗？删除后不可恢复。`)) return;
    setDeletingId(id);
    setDeleteError(null);
    const error = await onDelete(id);
    if (error) setDeleteError(error);
    setDeletingId(null);
  };

  if (opened) {
    return (
      <div>
        <div className="card">
          <div className="plan-row" style={{ border: "none", padding: 0 }}>
            <div>
              <div className="plan-name">{opened.name}</div>
              <div className="plan-meta">
                保存于 {opened.created_at} · 快照 {opened.snapshot_id} · {opened.operations.length} 笔操作
              </div>
            </div>
            <span className="spacer" />
            <button
              className="btn"
              onClick={() => {
                onLoadOps(opened.operations);
              }}
              title="将操作列表载入调仓模拟页作为新草稿（原方案保持不变）"
            >
              载入模拟器
            </button>
            <button className="btn" onClick={onClose}>
              返回方案列表
            </button>
          </div>
        </div>
        <div className="card">
          <h2>方案操作（只读）</h2>
          <div className="op-list">
            {opened.operations.map((op, index) => (
              <div className="op-row" key={index}>
                <span className="op-index">{index + 1}</span>
                <span className={`op-action ${op.action === "BUY" ? "buy" : "sell"}`}>
                  {op.action === "BUY" ? "买入" : "卖出"}
                </span>
                <span>
                  {op.symbol}
                  {opened.simulated.holdings.find((h) => h.symbol === op.symbol)?.name ?? ""}
                </span>
                <span className="op-nums">× {op.quantity}</span>
                <span className="op-nums">
                  {index < opened.applied.length ? `${opened.applied[index].amount} 元` : "—"}
                </span>
                <span className="op-nums">
                  {index < opened.applied.length ? `现金 ${opened.applied[index].cash_after}` : "—"}
                </span>
                <span className="op-status-tag">已生效</span>
              </div>
            ))}
          </div>
          <div className="hint">
            保存时已通过全部校验；重开结果由后端按操作列表对库内快照重新计算，原始组合不受影响。
            载入模拟器后生成的是新草稿，请以新名称保存，原方案不会被覆盖。
          </div>
        </div>
        <CompareTable
          orig={opened.original_analysis}
          sim={opened.analysis}
          simCash={opened.simulated.cash}
          appliedCount={opened.applied.length}
          error={opened.error}
          title={opened.name}
        />
      </div>
    );
  }

  return (
    <div className="card">
      <h2>已保存方案</h2>
      {openError && <div className="inline-error">⚠ {openError}</div>}
      {deleteError && <div className="inline-error">⚠ {deleteError}</div>}
      {plans.length === 0 ? (
        <div className="empty-state">
          暂无已保存方案。在「调仓模拟」页添加操作（或使用目标权重建议）、命名并保存后，可在此重新打开查看。
        </div>
      ) : (
        <div className="plan-list">
          {plans.map((plan) => (
            <div className="plan-row" key={plan.id}>
              <div>
                <div className="plan-name">{plan.name}</div>
                <div className="plan-meta">
                  #{plan.id} · 保存于 {plan.created_at} · 快照 {plan.snapshot_id}
                </div>
              </div>
              <span className="spacer" />
              <button className="btn btn-primary" onClick={() => onOpen(plan.id)}>
                打开
              </button>
              <button
                className="btn"
                disabled={deletingId === plan.id}
                onClick={() => void handleDelete(plan.id, plan.name)}
              >
                删除
              </button>
            </div>
          ))}
        </div>
      )}
      <div className="hint">
        方案保存在服务端 SQLite 中，刷新页面、重启服务后仍可重开；原始组合始终不变。
      </div>
    </div>
  );
}
