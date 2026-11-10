"""引擎数据类 -> API JSON 的组装层（所有数值均为后端预格式化字符串）。"""
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

from .engine import Analysis, InstrumentDef, PortfolioState, Rules, SimulationResult, analyze
from .money import format_cents, format_weight


def analysis_payload(analysis: Analysis) -> dict:
    return {
        "total_assets": analysis.total_assets,
        "instruments_value": analysis.instruments_value,
        "cash": analysis.cash,
        "cash_weight": analysis.cash_weight,
        "holdings_count": analysis.holdings_count,
        "items": [
            {
                "symbol": item.symbol,
                "name": item.name,
                "industry": item.industry,
                "quantity": item.quantity,
                "market_value": item.market_value,
                "weight": item.weight,
                "weight_percent": item.weight_percent,
                "industry_weight": item.industry_weight,
                "industry_weight_percent": item.industry_weight_percent,
                "single_triggered": item.single_triggered,
                "industry_triggered": item.industry_triggered,
            }
            for item in analysis.items
        ],
        "concentration": {
            "single_instrument": [
                {"symbol": item.target, "weight_percent": item.weight_percent, "threshold_percent": item.threshold_percent}
                for item in analysis.single_concentration
            ],
            "industry": [
                {"industry": item.target, "weight_percent": item.weight_percent, "threshold_percent": item.threshold_percent}
                for item in analysis.industry_concentration
            ],
        },
    }


def portfolio_payload(state: PortfolioState, instruments: dict[str, InstrumentDef], rules: Rules, analysis: Analysis) -> dict:
    return {
        "portfolio_id": state.portfolio_id,
        "name": state.name,
        "currency": state.currency,
        "snapshot_id": state.snapshot_id,
        "snapshot_at": state.snapshot_at,
        "cash": format_cents(state.cash_cents),
        "instruments": [
            {
                "symbol": d.symbol,
                "name": d.name,
                "industry": d.industry,
                "price": format_cents(d.price_cents),
                "quantity": state.holdings.get(d.symbol, 0),
            }
            for d in instruments.values()
        ],
        "analysis": analysis_payload(analysis),
        "rules": {
            "single_instrument_threshold": format_weight(rules.single_instrument_threshold)[1],
            "industry_threshold": format_weight(rules.industry_threshold)[1],
            "comparison": rules.comparison,
            "cash_in_denominator": rules.cash_in_denominator,
            "cash_in_industry": rules.cash_in_industry,
            "fees": format_cents(rules.fees_cents),
        },
    }


def format_signed_percent(value: Fraction) -> str:
    """带符号的两位小数百分比字符串，例如 1/300 -> "+0.33"、-1/300 -> "-0.33"。"""
    percent = (Decimal(value.numerator) / Decimal(value.denominator) * 100).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    sign = "+" if percent > 0 else ""
    return f"{sign}{percent.__format__('f')}"


def simulate_payload(
    result: SimulationResult,
    instruments: dict[str, InstrumentDef],
    rules: Rules,
    original_analysis: Analysis,
) -> dict:
    analysis = analyze(result.state, instruments, rules)
    return {
        "applied": [
            {
                "action": item.operation.action,
                "symbol": item.operation.symbol,
                "quantity": item.operation.quantity,
                "amount": format_cents(item.amount_cents),
                "cash_after": format_cents(item.cash_after_cents),
            }
            for item in result.applied
        ],
        "simulated": {
            "cash": format_cents(result.state.cash_cents),
            "holdings": [
                {"symbol": d.symbol, "name": d.name, "industry": d.industry,
                 "price": format_cents(d.price_cents), "quantity": result.state.holdings.get(d.symbol, 0)}
                for d in instruments.values()
            ],
        },
        "analysis": analysis_payload(analysis),
        "original_analysis": analysis_payload(original_analysis),
        "error": (
            {
                "op_index": result.error.op_index,
                "code": result.error.code,
                "message": result.error.message,
            }
            if result.error
            else None
        ),
    }
