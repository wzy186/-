"""POST /api/rebalance：目标权重调仓建议。

- mode=equal：全部标的等权（1/n），现金目标 0；
- mode=targets：自定义权重（未列出的标的保持当前持仓），权重之和 ≤ 1，现金为剩余。
建议由引擎生成并经与人工操作完全相同的顺序校验（先卖后买、现金不足自动减量），
返回建议操作、模拟结果与目标/实际偏差报告。
"""
from fractions import Fraction

from fastapi import APIRouter, HTTPException, Request

from ..engine import analyze, suggest_rebalance
from ..money import format_weight, parse_threshold, weight_fraction
from ..presenters import format_signed_percent, simulate_payload
from ..schemas import RebalanceRequest

router = APIRouter(prefix="/api", tags=["rebalance"])


@router.post("/rebalance")
def rebalance(request: Request, body: RebalanceRequest) -> dict:
    state, instruments, rules = request.app.state.portfolio

    if body.mode == "equal":
        share = Fraction(1, len(instruments))
        targets = {symbol: share for symbol in instruments}
    else:
        if not body.targets:
            raise HTTPException(
                status_code=422,
                detail={"code": "EMPTY_TARGETS", "message": "至少提供一个目标权重"},
            )
        targets: dict[str, Fraction] = {}
        for symbol, raw in body.targets.items():
            if symbol not in instruments:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "UNKNOWN_SYMBOL", "message": f"未知标的代码：{symbol}"},
                )
            try:
                weight = parse_threshold(raw)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "INVALID_TARGET", "message": f"目标权重格式错误：{symbol} = {raw}"},
                )
            if weight < 0 or weight > 1:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "INVALID_TARGET", "message": f"目标权重须在 0~1 之间：{symbol} = {raw}"},
                )
            targets[symbol] = weight
        if sum(targets.values()) > 1:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "INVALID_TARGETS",
                    "message": "目标权重之和不能超过 1（现金占比为剩余部分）",
                },
            )

    ops, result = suggest_rebalance(state, targets, instruments)
    original_analysis = analyze(state, instruments, rules)
    payload = simulate_payload(result, instruments, rules, original_analysis)
    analysis = analyze(result.state, instruments, rules)

    total = original_analysis.total_assets_cents
    cash_target = 1 - sum(targets.values())
    deviation = []
    for item in analysis.items:
        if item.symbol not in targets:
            continue
        actual = weight_fraction(item.market_value_cents, total)
        deviation.append(
            {
                "symbol": item.symbol,
                "name": item.name,
                "target_percent": format_weight(targets[item.symbol])[1],
                "actual_percent": item.weight_percent,
                "deviation_percent": format_signed_percent(actual - targets[item.symbol]),
            }
        )
    deviation.append(
        {
            "symbol": "CASH",
            "name": "现金",
            "target_percent": format_weight(cash_target)[1],
            "actual_percent": format_weight(weight_fraction(result.state.cash_cents, total))[1],
            "deviation_percent": format_signed_percent(
                weight_fraction(result.state.cash_cents, total) - cash_target
            ),
        }
    )

    payload["suggested_ops"] = [{"action": a, "symbol": s, "quantity": q} for a, s, q in ops]
    payload["targets"] = [
        {"symbol": symbol, "weight_percent": format_weight(targets[symbol])[1]}
        for symbol in instruments
        if symbol in targets
    ]
    payload["cash_target_percent"] = format_weight(cash_target)[1]
    payload["deviation"] = deviation
    return payload
