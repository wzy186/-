"""组合计算引擎：纯函数、无 I/O、不依赖 FastAPI。

全部业务规则在此实现，API 层只负责入参解析与响应组装，
保证前端展示与后端校验永远同一口径（FR-05）。

核心语义：
- 金额单位为「分」（int），占比为 Fraction 精确比较（见 money.py）；
- 模拟从原始组合（base）出发逐笔执行，后续操作使用前序结果；
- 遇错即停：第一笔失败的操作之后的操作不再尝试，返回最后有效状态；
- 所有状态均为 frozen dataclass，持仓字典 copy-on-write，原始组合天然隔离。
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Mapping, Optional, Sequence

from .money import (
    format_cents,
    format_weight,
    parse_cents,
    parse_threshold,
    weight_fraction,
    weight_gt,
)

# 数量上限：正整数数量为业务约束，此处仅防御极端输入
MAX_QUANTITY = 1_000_000_000


@dataclass(frozen=True)
class InstrumentDef:
    """标的静态定义（来自行情快照）：代码、名称、行业、固定价格（分）。"""

    symbol: str
    name: str
    industry: str
    price_cents: int


@dataclass(frozen=True)
class PortfolioState:
    """某一时点的组合状态：现金（分）与各标的持仓数量。"""

    portfolio_id: str
    name: str
    currency: str
    snapshot_id: str
    snapshot_at: str
    cash_cents: int
    holdings: Mapping[str, int]  # symbol -> quantity，操作时整体复制后替换


@dataclass(frozen=True)
class Operation:
    """一笔模拟操作：买入或卖出给定标的的正整数数量。"""

    action: str  # "BUY" | "SELL"
    symbol: str
    quantity: int


@dataclass(frozen=True)
class OpError:
    """一笔操作校验失败的原因（op_index 从 1 开始）。"""

    op_index: int
    code: str  # INVALID_QUANTITY / UNKNOWN_SYMBOL / EXCESS_QUANTITY / INSUFFICIENT_CASH
    message: str


@dataclass(frozen=True)
class Rules:
    """从种子数据 rules 块解析出的业务规则（不硬编码到计算逻辑之外）。"""

    single_instrument_threshold: Fraction
    industry_threshold: Fraction
    comparison: str  # 仅支持 strictly_greater_than
    cash_in_denominator: bool  # 仅支持 True
    cash_in_industry: bool  # 仅支持 False
    fees_cents: int  # 仅支持 0


@dataclass(frozen=True)
class HoldingAnalysis:
    """单个标的的分析结果：市值、占比、行业占比与触发标记（展示值均为字符串）。"""

    symbol: str
    name: str
    industry: str
    quantity: int
    price_cents: int
    market_value_cents: int
    market_value: str
    weight: str
    weight_percent: str
    industry_weight: str
    industry_weight_percent: str
    single_triggered: bool
    industry_triggered: bool


@dataclass(frozen=True)
class ConcentrationItem:
    """一条超限提醒：触发对象、实际比例与适用阈值（百分比展示字符串）。"""

    target: str  # 标的代码或行业名称
    weight_percent: str
    threshold_percent: str


@dataclass(frozen=True)
class Analysis:
    """组合分析结果：同时保留「分」级精确值与展示字符串，便于测试断言。"""

    total_assets_cents: int
    instruments_value_cents: int
    cash_cents: int
    holdings_count: int
    total_assets: str
    instruments_value: str
    cash: str
    cash_weight: str
    items: tuple[HoldingAnalysis, ...]
    single_concentration: tuple[ConcentrationItem, ...]
    industry_concentration: tuple[ConcentrationItem, ...]


@dataclass(frozen=True)
class AppliedOp:
    """一笔已生效操作及其影响：成交金额与执行后的现金（分）。"""

    operation: Operation
    amount_cents: int
    cash_after_cents: int


@dataclass(frozen=True)
class SimulationResult:
    """顺序模拟结果：已生效操作前缀、最后有效状态与失败原因（若有）。"""

    applied: tuple[AppliedOp, ...]
    state: PortfolioState  # 最后有效状态；第一笔即失败时等于 base
    error: Optional[OpError]


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def parse_portfolio(payload: Mapping) -> tuple[PortfolioState, dict[str, InstrumentDef], Rules]:
    """解析种子 JSON 为组合状态、标的定义与业务规则，并对结构做校验。"""
    state = PortfolioState(
        portfolio_id=_require_str(payload, "portfolio_id"),
        name=_require_str(payload, "name"),
        currency=_require_str(payload, "currency"),
        snapshot_id=_require_str(payload, "snapshot_id"),
        snapshot_at=_require_str(payload, "snapshot_at"),
        cash_cents=parse_cents(_require_str(payload, "cash")),
        holdings={},
    )

    raw_instruments = payload.get("instruments")
    if not isinstance(raw_instruments, list) or not raw_instruments:
        raise ValueError("instruments 必须为非空数组")
    instruments: dict[str, InstrumentDef] = {}
    holdings: dict[str, int] = {}
    for raw in raw_instruments:
        symbol = _require_str(raw, "symbol")
        if symbol in instruments:
            raise ValueError(f"标的代码重复：{symbol}")
        quantity = raw.get("quantity")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0:
            raise ValueError(f"{symbol} 的持仓数量必须为非负整数：{quantity!r}")
        instruments[symbol] = InstrumentDef(
            symbol=symbol,
            name=_require_str(raw, "name"),
            industry=_require_str(raw, "industry"),
            price_cents=parse_cents(_require_str(raw, "price")),
        )
        holdings[symbol] = quantity
    state = replace(state, holdings=holdings)

    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, Mapping):
        raise ValueError("rules 必须为对象")
    rules = Rules(
        single_instrument_threshold=parse_threshold(_require_str(raw_rules, "single_instrument_threshold")),
        industry_threshold=parse_threshold(_require_str(raw_rules, "industry_threshold")),
        comparison=_require_str(raw_rules, "comparison"),
        cash_in_denominator=bool(raw_rules.get("cash_in_denominator")),
        cash_in_industry=bool(raw_rules.get("cash_in_industry")),
        fees_cents=parse_cents(_require_str(raw_rules, "fees")),
    )
    if rules.comparison != "strictly_greater_than":
        raise ValueError(f"仅支持 strictly_greater_than 比较方式：{rules.comparison}")
    if not rules.cash_in_denominator:
        raise ValueError("仅支持现金计入总资产分母（cash_in_denominator=true）")
    if rules.cash_in_industry:
        raise ValueError("仅支持现金不归入行业（cash_in_industry=false）")
    if rules.fees_cents != 0:
        raise ValueError(f"本次范围仅支持手续费为 0，实际为 {format_cents(rules.fees_cents)} 元")
    return state, instruments, rules


def _require_str(mapping: Mapping, key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 必须为非空字符串：{value!r}")
    return value.strip()


# ---------------------------------------------------------------------------
# 操作执行
# ---------------------------------------------------------------------------


def apply_op(
    state: PortfolioState,
    action: object,
    symbol: object,
    quantity: object,
    instruments: Mapping[str, InstrumentDef],
) -> tuple[PortfolioState, Optional[OpError]]:
    """对组合状态执行一笔操作，返回新状态；校验失败返回原状态与错误。

    校验顺序：数量 -> 标的代码 -> 卖出持仓 -> 买入现金。
    成功后返回 copy-on-write 的新状态，原状态对象不受影响。
    """
    if action not in ("BUY", "SELL"):
        return state, OpError(1, "INVALID_ACTION", f"不支持的交易方向：{action!r}")

    qty = _validate_quantity(quantity)
    if qty is None:
        return state, OpError(1, "INVALID_QUANTITY", f"交易数量必须为正整数：{quantity!r}")

    instrument = instruments.get(str(symbol)) if isinstance(symbol, str) else None
    if instrument is None:
        return state, OpError(1, "UNKNOWN_SYMBOL", f"未知标的代码：{symbol!r}")

    holdings = dict(state.holdings)
    current = holdings.get(instrument.symbol, 0)

    if action == "SELL":
        if qty > current:
            return state, OpError(
                1,
                "EXCESS_QUANTITY",
                f"卖出数量超过当前模拟持仓：卖出 {instrument.symbol} × {qty}，当前持仓 {current}",
            )
        holdings[instrument.symbol] = current - qty
        return replace(state, cash_cents=state.cash_cents + instrument.price_cents * qty, holdings=holdings), None

    cost_cents = instrument.price_cents * qty
    if cost_cents > state.cash_cents:
        return state, OpError(
            1,
            "INSUFFICIENT_CASH",
            f"买入 {instrument.symbol} × {qty} 需要 {format_cents(cost_cents)} 元，现金仅 {format_cents(state.cash_cents)} 元",
        )
    holdings[instrument.symbol] = current + qty
    return replace(state, cash_cents=state.cash_cents - cost_cents, holdings=holdings), None


def _validate_quantity(quantity: object) -> Optional[int]:
    """正整数校验：接受 int 与整数值 float（如 2.0），拒绝 bool/零/负数/小数。"""
    if isinstance(quantity, bool):
        return None
    if isinstance(quantity, int):
        return quantity if 0 < quantity <= MAX_QUANTITY else None
    if isinstance(quantity, float):
        if quantity.is_integer() and 0 < quantity <= MAX_QUANTITY:
            return int(quantity)
    return None


def simulate_sequential(
    base: PortfolioState,
    ops: Sequence[tuple[object, object, object]],
    instruments: Mapping[str, InstrumentDef],
) -> SimulationResult:
    """从原始组合出发按顺序执行操作，遇错即停。

    ops 元素为 (action, symbol, quantity) 原始值，数量校验在引擎内完成，
    保证所有非法输入（含小数数量）返回同一错误结构。
    """
    state = base
    applied: list[AppliedOp] = []
    for index, (action, symbol, quantity) in enumerate(ops, start=1):
        new_state, error = apply_op(state, action, symbol, quantity, instruments)
        if error is not None:
            return SimulationResult(
                applied=tuple(applied),
                state=state,  # 最后有效状态（失败的操作未生效）
                error=OpError(op_index=index, code=error.code, message=f"第 {index} 笔操作失败：{error.message}"),
            )
        operation = Operation(action=str(action), symbol=str(symbol), quantity=int(quantity))
        applied.append(
            AppliedOp(
                operation=operation,
                amount_cents=instruments[operation.symbol].price_cents * operation.quantity,
                cash_after_cents=new_state.cash_cents,
            )
        )
        state = new_state
    return SimulationResult(applied=tuple(applied), state=state, error=None)


# ---------------------------------------------------------------------------
# 目标权重调仓建议
# ---------------------------------------------------------------------------


def suggest_rebalance(
    base: PortfolioState,
    targets: Mapping[str, Fraction],
    instruments: Mapping[str, InstrumentDef],
) -> tuple[list[tuple[str, str, int]], SimulationResult]:
    """按目标权重生成调仓建议（买卖操作列表）。

    算法（确定性，全部整数分/Fraction 运算）：
    1. 目标市值 = 总资产 × 目标权重；目标数量 = round_half_up(目标市值 / 价格)；
       未出现在 targets 中的标的保持当前持仓不变；
    2. 生成「先卖出、后买入」的操作序列（卖出释放的现金可供后续买入使用）；
    3. 复用 simulate_sequential 做与人工操作完全相同的校验；
    4. 若因整数取整导致现金不足：从最后一笔买入开始逐股减量
       （减到 0 则放弃该笔买入）直至可行，减量偏差体现在偏差报告中。
    """
    total = base.cash_cents + sum(
        d.price_cents * base.holdings.get(s, 0) for s, d in instruments.items()
    )
    sells: list[tuple[str, int]] = []
    buys: list[tuple[str, int]] = []
    for symbol, d in instruments.items():
        if symbol in targets:
            target_value = targets[symbol] * total
            target_qty = round_half_up(Fraction(target_value, d.price_cents))
        else:
            target_qty = base.holdings.get(symbol, 0)
        diff = target_qty - base.holdings.get(symbol, 0)
        if diff > 0:
            buys.append((symbol, diff))
        elif diff < 0:
            sells.append((symbol, -diff))

    ops: list[tuple[str, str, int]] = [("SELL", s, q) for s, q in sells] + [
        ("BUY", s, q) for s, q in buys
    ]
    result = simulate_sequential(base, ops, instruments)
    while result.error is not None and result.error.code == "INSUFFICIENT_CASH":
        adjusted = False
        for i in range(len(ops) - 1, -1, -1):
            action, symbol, qty = ops[i]
            if action == "BUY" and qty > 1:
                ops[i] = (action, symbol, qty - 1)
                adjusted = True
                break
            if action == "BUY" and qty == 1:
                ops.pop(i)
                adjusted = True
                break
        if not adjusted:
            break
        result = simulate_sequential(base, ops, instruments)
    return ops, result


def round_half_up(value: Fraction) -> int:
    """四舍五入（半入）到整数：floor(value + 1/2)。"""
    return int(value + Fraction(1, 2))


# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------


def analyze(state: PortfolioState, instruments: Mapping[str, InstrumentDef], rules: Rules) -> Analysis:
    """计算组合概览与集中度：市值、占比、行业分布与超限触发列表。"""
    instruments_value = 0
    industry_value: dict[str, int] = {}
    items: list[HoldingAnalysis] = []

    for instrument in instruments.values():
        quantity = state.holdings.get(instrument.symbol, 0)
        market_value = instrument.price_cents * quantity
        instruments_value += market_value
        industry_value[instrument.industry] = industry_value.get(instrument.industry, 0) + market_value
        items.append((instrument, quantity, market_value))

    total = instruments_value + state.cash_cents  # cash_in_denominator=True
    single_threshold = rules.single_instrument_threshold
    industry_threshold = rules.industry_threshold

    holding_items: list[HoldingAnalysis] = []
    single_concentration: list[ConcentrationItem] = []
    industry_concentration: list[ConcentrationItem] = []
    holdings_count = 0

    for instrument, quantity, market_value in items:
        weight = weight_fraction(market_value, total)
        industry_weight = weight_fraction(industry_value[instrument.industry], total)
        single_triggered = weight_gt(weight, single_threshold)
        industry_triggered = weight_gt(industry_weight, industry_threshold)
        if quantity > 0:
            holdings_count += 1
        weight_str, weight_percent = format_weight(weight)
        ind_weight_str, ind_weight_percent = format_weight(industry_weight)
        holding_items.append(
            HoldingAnalysis(
                symbol=instrument.symbol,
                name=instrument.name,
                industry=instrument.industry,
                quantity=quantity,
                price_cents=instrument.price_cents,
                market_value_cents=market_value,
                market_value=format_cents(market_value),
                weight=weight_str,
                weight_percent=weight_percent,
                industry_weight=ind_weight_str,
                industry_weight_percent=ind_weight_percent,
                single_triggered=single_triggered,
                industry_triggered=industry_triggered,
            )
        )
        if single_triggered:
            single_concentration.append(
                ConcentrationItem(
                    target=instrument.symbol,
                    weight_percent=weight_percent,
                    threshold_percent=format_weight(single_threshold)[1],
                )
            )

    # 行业触发去重：同一行业的多个标的共用同一行业占比，仅记录一条
    seen_industries: set[str] = set()
    for instrument, _, _ in items:
        industry = instrument.industry
        if industry in seen_industries:
            continue
        seen_industries.add(industry)
        industry_weight = weight_fraction(industry_value[industry], total)
        if weight_gt(industry_weight, industry_threshold):
            industry_concentration.append(
                ConcentrationItem(
                    target=industry,
                    weight_percent=format_weight(industry_weight)[1],
                    threshold_percent=format_weight(industry_threshold)[1],
                )
            )

    cash_weight, _ = format_weight(weight_fraction(state.cash_cents, total))
    return Analysis(
        total_assets_cents=total,
        instruments_value_cents=instruments_value,
        cash_cents=state.cash_cents,
        holdings_count=holdings_count,
        total_assets=format_cents(total),
        instruments_value=format_cents(instruments_value),
        cash=format_cents(state.cash_cents),
        cash_weight=cash_weight,
        items=tuple(holding_items),
        single_concentration=tuple(single_concentration),
        industry_concentration=tuple(industry_concentration),
    )
