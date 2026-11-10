/** URL 验证辅助参数解析（纯函数，便于单元测试）。

    - ops=SELL:A01:100,BUY:A02:100  预填操作列表
    - plan=3                        打开指定方案
    - rebalance=equal               自动生成等权分散建议
    - tab=overview|simulate|plans   指定初始页签
*/
import type { Op } from "./types";

export type Tab = "overview" | "simulate" | "plans";

export function parseOpsParam(search: string): Op[] {
  const raw = new URLSearchParams(search).get("ops");
  if (!raw) return [];
  const ops: Op[] = [];
  for (const part of raw.split(",")) {
    const [action, symbol, quantity] = part.split(":");
    if (
      (action === "BUY" || action === "SELL") &&
      symbol &&
      Number.isInteger(Number(quantity)) &&
      Number(quantity) > 0
    ) {
      ops.push({ action, symbol, quantity: Number(quantity) });
    }
  }
  return ops;
}

export function parsePlanParam(search: string): number | null {
  const raw = new URLSearchParams(search).get("plan");
  const id = Number(raw);
  return raw && Number.isInteger(id) && id > 0 ? id : null;
}

export function parseRebalanceParam(search: string): "equal" | null {
  const raw = new URLSearchParams(search).get("rebalance");
  return raw === "equal" ? "equal" : null;
}

export function initialTab(search: string, ops: Op[], planId: number | null): Tab {
  const param = new URLSearchParams(search).get("tab");
  if (param === "overview" || param === "simulate" || param === "plans") return param;
  if (ops.length > 0) return "simulate";
  if (planId !== null) return "plans";
  return "overview";
}
