/** 应用主控：加载组合、编排三个页签与模拟/保存/重开流程。

    URL 参数（验证辅助入口，便于复现验收场景与截图，解析逻辑见 urlParams.ts）：
    - ?tab=overview|simulate|plans  指定初始页签
    - ?ops=SELL:A01:100,BUY:A02:100 预填操作列表并进入模拟页
    - ?plan=3                       进入方案页并打开指定方案
    - ?rebalance=equal              进入模拟页并自动生成等权分散建议
*/
import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api";
import CompareTable from "./components/CompareTable";
import ConcentrationPanel from "./components/ConcentrationPanel";
import ErrorBanner from "./components/ErrorBanner";
import Overview from "./components/Overview";
import RebalancePanel from "./components/RebalancePanel";
import SavedPlansPanel from "./components/SavedPlansPanel";
import SimulationPanel from "./components/SimulationPanel";
import type { Op, PlanDetail, PlanSummary, Portfolio, SimulateResponse } from "./types";
import { initialTab, parseOpsParam, parsePlanParam, parseRebalanceParam, Tab } from "./urlParams";

const TAB_LABELS: { key: Tab; label: string }[] = [
  { key: "overview", label: "组合概览" },
  { key: "simulate", label: "调仓模拟" },
  { key: "plans", label: "已存方案" },
];

export default function App() {
  const search = useRef(window.location.search).current;
  const initialOps = useRef(parseOpsParam(search)).current;
  const initialPlan = useRef(parsePlanParam(search)).current;
  const initialRebalance = useRef(parseRebalanceParam(search)).current;

  const [tab, setTab] = useState<Tab>(() => initialTab(search, initialOps, initialPlan));
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [ops, setOps] = useState<Op[]>(initialOps);
  const [simResult, setSimResult] = useState<SimulateResponse | null>(null);
  const [simError, setSimError] = useState<string | null>(null);
  const simSeq = useRef(0);

  const [plans, setPlans] = useState<PlanSummary[]>([]);
  const [openedPlan, setOpenedPlan] = useState<PlanDetail | null>(null);
  const [openError, setOpenError] = useState<string | null>(null);

  const [saveName, setSaveName] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveOk, setSaveOk] = useState<string | null>(null);

  const loadPortfolio = useCallback(async () => {
    setLoadError(null);
    try {
      const data = await api.getPortfolio();
      setPortfolio(data);
    } catch (e) {
      setLoadError(e instanceof ApiError ? e.message : "组合加载失败，请确认后端服务已启动");
    }
  }, []);

  const refreshPlans = useCallback(async () => {
    try {
      const data = await api.listPlans();
      setPlans(data.plans);
    } catch {
      /* 方案列表加载失败不阻塞主流程 */
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    loadPortfolio();
    refreshPlans();
    return () => controller.abort();
  }, [loadPortfolio, refreshPlans]);

  // 操作列表变化 -> 防抖 250ms 调后端模拟（打开方案期间暂停）
  useEffect(() => {
    if (openedPlan) return;
    if (ops.length === 0) {
      setSimResult(null);
      setSimError(null);
      return;
    }
    const seq = ++simSeq.current;
    const timer = setTimeout(async () => {
      try {
        const result = await api.simulate(ops);
        if (seq === simSeq.current) {
          setSimResult(result);
          setSimError(null);
        }
      } catch (e) {
        if (seq === simSeq.current) {
          setSimError(e instanceof ApiError ? e.message : "模拟请求失败，请稍后重试");
        }
      }
    }, 250);
    return () => clearTimeout(timer);
  }, [ops, openedPlan]);

  // ?plan= 参数：进入方案页并打开
  useEffect(() => {
    if (initialPlan !== null) {
      openPlan(initialPlan);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const openPlan = useCallback(async (id: number) => {
    setOpenError(null);
    try {
      const detail = await api.getPlan(id);
      setOpenedPlan(detail);
      setTab("plans");
    } catch (e) {
      setOpenError(e instanceof ApiError ? e.message : "方案打开失败");
    }
  }, []);

  const closePlan = useCallback(() => {
    setOpenedPlan(null);
  }, []);

  const addOp = useCallback((op: Op) => {
    setOps((prev) => [...prev, op]);
    setSaveOk(null);
    setSaveError(null);
  }, []);

  const removeOp = useCallback((index: number) => {
    setOps((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const moveOp = useCallback((index: number, dir: -1 | 1) => {
    setOps((prev) => {
      const next = [...prev];
      const target = index + dir;
      if (target < 0 || target >= next.length) return prev;
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  }, []);

  const savePlan = useCallback(async () => {
    setSaving(true);
    setSaveError(null);
    setSaveOk(null);
    try {
      const saved = await api.createPlan(saveName.trim(), ops);
      setSaveOk(`方案「${saved.name}」已保存，可在「已存方案」页重开`);
      setSaveName("");
      await refreshPlans();
    } catch (e) {
      setSaveError(e instanceof ApiError ? e.message : "保存失败，请稍后重试");
    } finally {
      setSaving(false);
    }
  }, [saveName, ops, refreshPlans]);

  const deletePlan = useCallback(
    async (id: number): Promise<string | null> => {
      try {
        await api.deletePlan(id);
        await refreshPlans();
        return null;
      } catch (e) {
        return e instanceof ApiError ? e.message : "删除失败，请稍后重试";
      }
    },
    [refreshPlans]
  );

  /** 目标权重建议 / 已存方案 -> 载入模拟器为新草稿（原方案保持不变） */
  const loadOps = useCallback((newOps: Op[]) => {
    setOps(newOps);
    setOpenedPlan(null);
    setSaveOk(null);
    setSaveError(null);
    setSaveName("");
    setTab("simulate");
  }, []);

  if (loadError) {
    return (
      <div className="app">
        <div className="app-header">
          <h1>投资组合分析与调仓模拟</h1>
        </div>
        <ErrorBanner message={loadError} onRetry={loadPortfolio} />
        <div className="empty-state">无法加载组合数据。请确认后端服务已启动（默认 http://127.0.0.1:8000）。</div>
      </div>
    );
  }

  if (!portfolio) {
    return (
      <div className="app">
        <div className="app-header">
          <h1>投资组合分析与调仓模拟</h1>
        </div>
        <div className="stat-tiles">
          <div className="stat-tile hero"><div className="skeleton" style={{ height: 44 }} /></div>
          <div className="stat-tile"><div className="skeleton" style={{ height: 44 }} /></div>
          <div className="stat-tile"><div className="skeleton" style={{ height: 44 }} /></div>
          <div className="stat-tile"><div className="skeleton" style={{ height: 44 }} /></div>
        </div>
        <div className="card"><div className="skeleton" style={{ height: 200 }} /></div>
      </div>
    );
  }

  const simAnalysis = simResult ? simResult.analysis : null;
  const simCash = simResult ? simResult.simulated.cash : null;
  const alertCount =
    portfolio.analysis.concentration.single_instrument.length +
    portfolio.analysis.concentration.industry.length;

  const tabBadge = (key: Tab): string | null => {
    if (key === "overview" && alertCount > 0) return `⚠${alertCount}`;
    if (key === "simulate" && ops.length > 0) return String(ops.length);
    if (key === "plans" && plans.length > 0) return String(plans.length);
    return null;
  };

  return (
    <div className="app">
      <div className="app-header">
        <h1>投资组合分析与调仓模拟</h1>
        <span className="subtitle">
          {portfolio.name}（{portfolio.portfolio_id}）
        </span>
        <span className="snapshot-chip">
          行情快照 {portfolio.snapshot_at} · {portfolio.currency}
        </span>
        <span className={`badge ${alertCount > 0 ? "badge-alert" : "badge-ok"}`}>
          {alertCount > 0 ? `⚠ ${alertCount} 项超限` : "✓ 无超限"}
        </span>
      </div>

      <div className="tabs">
        {TAB_LABELS.map(({ key, label }) => (
          <button key={key} className={tab === key ? "active" : ""} onClick={() => setTab(key)}>
            {label}
            {tabBadge(key) && <span className="tab-badge">{tabBadge(key)}</span>}
          </button>
        ))}
      </div>

      {tab === "overview" && (
        <div>
          <Overview portfolio={portfolio} />
          <ConcentrationPanel analysis={portfolio.analysis} rules={portfolio.rules} />
        </div>
      )}

      {tab === "simulate" && (
        <div className="section-row">
          <div>
            <SimulationPanel
              instruments={portfolio.instruments}
              ops={ops}
              applied={simResult?.applied ?? []}
              error={simResult?.error ?? null}
              onAdd={addOp}
              onRemove={removeOp}
              onMove={moveOp}
              onClear={() => {
                setOps([]);
                setSaveOk(null);
              }}
              saveName={saveName}
              onSaveNameChange={setSaveName}
              onSave={savePlan}
              saving={saving}
              saveError={saveError}
              saveOk={saveOk}
            />
            <RebalancePanel
              instruments={portfolio.instruments}
              currentOpsCount={ops.length}
              onLoadOps={loadOps}
              autoGenerate={initialRebalance}
            />
          </div>
          <div>
            {simError && <ErrorBanner message={simError} />}
            <CompareTable
              orig={portfolio.analysis}
              sim={simAnalysis}
              simCash={simCash}
              appliedCount={simResult?.applied.length ?? 0}
              error={simResult?.error ?? null}
            />
          </div>
        </div>
      )}

      {tab === "plans" && (
        <SavedPlansPanel
          plans={plans}
          opened={openedPlan}
          openError={openError}
          onOpen={openPlan}
          onClose={closePlan}
          onDelete={deletePlan}
          onLoadOps={loadOps}
        />
      )}

      <div className="app-footer">
        本应用为技术测评原型：全部数据均为虚构测评数据，不构成任何投资建议；仅支持给定标的与固定行情快照价格，
        不涉及手续费（当前规则为 0）、税费、分红、融资、卖空或真实下单。
      </div>
    </div>
  );
}
