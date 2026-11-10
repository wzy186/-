/** 类型化 API 封装与错误信息提取（兼容业务错误与 FastAPI 校验错误两种结构）。 */
import type { Op, PlanDetail, PlanSummary, Portfolio, RebalanceResponse, SimulateResponse } from "./types";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

function extractMessage(detail: unknown): string {
  // 业务错误：{"detail": {"code": ..., "message": ...}}
  if (detail && typeof detail === "object" && "message" in (detail as Record<string, unknown>)) {
    return String((detail as Record<string, unknown>).message);
  }
  // FastAPI 结构校验错误：{"detail": [{msg, ...}, ...]}
  if (Array.isArray(detail)) {
    const parts = detail
      .map((d) => (d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : ""))
      .filter(Boolean);
    if (parts.length) return parts.join("；");
  }
  return "请求失败，请稍后重试";
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, {
    ...init,
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
  });
  if (!resp.ok) {
    let detail: unknown = null;
    try {
      detail = (await resp.json()).detail;
    } catch {
      /* 非 JSON 响应保持默认错误信息 */
    }
    throw new ApiError(extractMessage(detail), resp.status);
  }
  if (resp.status === 204) return undefined as T;
  return resp.json() as Promise<T>;
}

export const api = {
  getPortfolio: () => request<Portfolio>("/api/portfolio"),
  simulate: (operations: Op[]) =>
    request<SimulateResponse>("/api/simulate", {
      method: "POST",
      body: JSON.stringify({ operations }),
    }),
  createPlan: (name: string, operations: Op[]) =>
    request<PlanSummary>("/api/plans", {
      method: "POST",
      body: JSON.stringify({ name, operations }),
    }),
  listPlans: () => request<{ plans: PlanSummary[] }>("/api/plans"),
  getPlan: (id: number) => request<PlanDetail>(`/api/plans/${id}`),
  deletePlan: (id: number) =>
    request<undefined>(`/api/plans/${id}`, { method: "DELETE" }),
  rebalance: (body: { mode: "equal" } | { mode: "targets"; targets: Record<string, string> }) =>
    request<RebalanceResponse>("/api/rebalance", { method: "POST", body: JSON.stringify(body) }),
};
