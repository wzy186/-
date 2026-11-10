/** 与后端 API JSON 一一对应的类型定义（后端返回的数值均为已格式化字符串）。 */

export interface HoldingItem {
  symbol: string;
  name: string;
  industry: string;
  quantity: number;
  market_value: string;
  weight: string;
  weight_percent: string;
  industry_weight: string;
  industry_weight_percent: string;
  single_triggered: boolean;
  industry_triggered: boolean;
}

export interface ConcentrationSingle {
  symbol: string;
  weight_percent: string;
  threshold_percent: string;
}

export interface ConcentrationIndustry {
  industry: string;
  weight_percent: string;
  threshold_percent: string;
}

export interface Analysis {
  total_assets: string;
  instruments_value: string;
  cash: string;
  cash_weight: string;
  holdings_count: number;
  items: HoldingItem[];
  concentration: {
    single_instrument: ConcentrationSingle[];
    industry: ConcentrationIndustry[];
  };
}

export interface Instrument {
  symbol: string;
  name: string;
  industry: string;
  price: string;
  quantity: number;
}

export interface Portfolio {
  portfolio_id: string;
  name: string;
  currency: string;
  snapshot_id: string;
  snapshot_at: string;
  cash: string;
  instruments: Instrument[];
  analysis: Analysis;
  rules: {
    single_instrument_threshold: string;
    industry_threshold: string;
    comparison: string;
    cash_in_denominator: boolean;
    cash_in_industry: boolean;
    fees: string;
  };
}

export interface Op {
  action: "BUY" | "SELL";
  symbol: string;
  quantity: number;
}

export interface AppliedOp extends Op {
  amount: string;
  cash_after: string;
}

export interface SimError {
  op_index: number;
  code: string;
  message: string;
}

export interface SimulateResponse {
  applied: AppliedOp[];
  simulated: {
    cash: string;
    holdings: Instrument[];
  };
  analysis: Analysis;
  original_analysis: Analysis;
  error: SimError | null;
}

export interface PlanSummary {
  id: number;
  name: string;
  snapshot_id: string;
  created_at: string;
}

export interface PlanDetail extends SimulateResponse {
  id: number;
  name: string;
  created_at: string;
  snapshot_id: string;
  operations: Op[];
}

export interface DeviationRow {
  symbol: string;
  name: string;
  target_percent: string;
  actual_percent: string;
  deviation_percent: string;
}

export interface RebalanceTarget {
  symbol: string;
  weight_percent: string;
}

export interface RebalanceResponse extends SimulateResponse {
  suggested_ops: Op[];
  targets: RebalanceTarget[];
  cash_target_percent: string;
  deviation: DeviationRow[];
}
