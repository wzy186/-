/** URL 参数解析单元测试（验收辅助入口的行为约定）。 */
import { describe, expect, it } from "vitest";
import { initialTab, parseOpsParam, parsePlanParam, parseRebalanceParam } from "./urlParams";

describe("parseOpsParam", () => {
  it("解析多笔操作", () => {
    expect(parseOpsParam("?ops=SELL:A01:100,BUY:A02:100")).toEqual([
      { action: "SELL", symbol: "A01", quantity: 100 },
      { action: "BUY", symbol: "A02", quantity: 100 },
    ]);
  });

  it("缺省时返回空列表", () => {
    expect(parseOpsParam("")).toEqual([]);
    expect(parseOpsParam("?tab=overview")).toEqual([]);
  });

  it("非法条目被跳过", () => {
    expect(parseOpsParam("?ops=SELL:A01:100,HOLD:A02:1,BUY:A01:2.5,BUY:A02:0")).toEqual([
      { action: "SELL", symbol: "A01", quantity: 100 },
    ]);
  });
});

describe("parsePlanParam", () => {
  it("解析正整数", () => {
    expect(parsePlanParam("?plan=3")).toBe(3);
  });

  it("非法值返回 null", () => {
    expect(parsePlanParam("")).toBeNull();
    expect(parsePlanParam("?plan=0")).toBeNull();
    expect(parsePlanParam("?plan=-1")).toBeNull();
    expect(parsePlanParam("?plan=1.5")).toBeNull();
    expect(parsePlanParam("?plan=abc")).toBeNull();
  });
});

describe("parseRebalanceParam", () => {
  it("仅接受 equal", () => {
    expect(parseRebalanceParam("?rebalance=equal")).toBe("equal");
    expect(parseRebalanceParam("?rebalance=targets")).toBeNull();
    expect(parseRebalanceParam("")).toBeNull();
  });
});

describe("initialTab", () => {
  const ops = [{ action: "SELL" as const, symbol: "A01", quantity: 100 }];

  it("显式 tab 参数优先", () => {
    expect(initialTab("?tab=plans", ops, 1)).toBe("plans");
    expect(initialTab("?tab=overview", ops, 1)).toBe("overview");
  });

  it("有操作时进入模拟页", () => {
    expect(initialTab("", ops, null)).toBe("simulate");
  });

  it("有方案时进入方案页", () => {
    expect(initialTab("", [], 1)).toBe("plans");
  });

  it("默认概览页", () => {
    expect(initialTab("", [], null)).toBe("overview");
  });
});
