"""计算引擎单元测试：逐条映射任务书验收用例 AC-01 ~ AC-04 及边界场景。"""
from fractions import Fraction

import pytest

from app.engine import (
    InstrumentDef,
    PortfolioState,
    analyze,
    parse_portfolio,
    simulate_sequential,
)
from app.money import format_cents


def total_cents(state: PortfolioState, instruments: dict[str, InstrumentDef]) -> int:
    """独立复算总资产（分）：持仓市值之和 + 现金。"""
    return state.cash_cents + sum(
        d.price_cents * state.holdings.get(s, 0) for s, d in instruments.items()
    )


class TestAC01InitialPortfolio:
    """初始组合：总资产 100,000 元；A01 占比 40%、科技 60%，均触发；其他对象不超限。"""

    def test_initial_state(self, base_portfolio):
        state, instruments, rules = base_portfolio
        analysis = analyze(state, instruments, rules)
        assert analysis.total_assets_cents == 10_000_000
        assert analysis.cash == "20000.00"
        assert analysis.instruments_value == "80000.00"
        assert analysis.total_assets == "100000.00"
        assert analysis.holdings_count == 6

    def test_a01_and_tech_trigger(self, base_portfolio):
        state, instruments, rules = base_portfolio
        analysis = analyze(state, instruments, rules)
        a01 = next(i for i in analysis.items if i.symbol == "A01")
        assert a01.market_value == "40000.00"
        assert a01.weight_percent == "40.00"
        assert a01.single_triggered is True
        assert a01.industry_weight_percent == "60.00"
        assert a01.industry_triggered is True
        assert [c.target for c in analysis.single_concentration] == ["A01"]
        assert [c.target for c in analysis.industry_concentration] == ["科技"]

    def test_others_not_triggered(self, base_portfolio):
        state, instruments, rules = base_portfolio
        analysis = analyze(state, instruments, rules)
        for item in analysis.items:
            if item.symbol == "A01":
                continue
            assert item.weight_percent == {"A02": "20.00", "A03": "10.00", "A04": "5.00",
                                           "A05": "3.00", "A06": "2.00"}[item.symbol]
            assert item.single_triggered is False


class TestAC02ThresholdBoundary:
    """卖出 A01×100 后：现金 30,000、A01 市值 30,000、总资产不变；
    A01 占比恰为 30%、科技恰为 50%，两项提醒均解除（等于不触发）。"""

    def test_sell_a01_100(self, base_portfolio):
        state, instruments, rules = base_portfolio
        result = simulate_sequential(state, [("SELL", "A01", 100)], instruments)
        assert result.error is None
        assert len(result.applied) == 1
        assert result.state.cash_cents == 3_000_000
        assert result.state.holdings["A01"] == 300
        assert total_cents(result.state, instruments) == 10_000_000

    def test_equality_clears_both_alerts(self, base_portfolio):
        state, instruments, rules = base_portfolio
        result = simulate_sequential(state, [("SELL", "A01", 100)], instruments)
        analysis = analyze(result.state, instruments, rules)
        a01 = next(i for i in analysis.items if i.symbol == "A01")
        assert a01.weight_percent == "30.00"
        assert a01.single_triggered is False
        assert a01.industry_weight_percent == "50.00"
        assert a01.industry_triggered is False
        assert analysis.single_concentration == ()
        assert analysis.industry_concentration == ()


class TestAC03SequentialDependency:
    """接 AC-02 再买入 A02×100：现金 25,000；A02 数量 500、市值 25,000；
    科技占比 55% 再次触发；总资产不变。"""

    def test_two_op_sequence(self, base_portfolio):
        state, instruments, rules = base_portfolio
        result = simulate_sequential(
            state, [("SELL", "A01", 100), ("BUY", "A02", 100)], instruments
        )
        assert result.error is None
        assert len(result.applied) == 2
        assert result.state.cash_cents == 2_500_000
        assert result.state.holdings["A02"] == 500
        assert total_cents(result.state, instruments) == 10_000_000

        analysis = analyze(result.state, instruments, rules)
        a02 = next(i for i in analysis.items if i.symbol == "A02")
        assert a02.market_value == "25000.00"
        assert a02.industry_weight_percent == "55.00"
        assert a02.industry_triggered is True
        assert [c.target for c in analysis.industry_concentration] == ["科技"]

    def test_second_op_uses_first_result(self, base_portfolio):
        """第二笔买入 A01×350 依赖第一笔卖出释放的现金：单独执行现金不足，按序执行成功。"""
        state, instruments, _ = base_portfolio
        alone = simulate_sequential(state, [("BUY", "A01", 350)], instruments)
        assert alone.error is not None
        assert alone.error.code == "INSUFFICIENT_CASH"  # 需 35,000 > 现金 20,000
        assert alone.state == state

        result = simulate_sequential(
            state, [("SELL", "A01", 200), ("BUY", "A01", 350)], instruments
        )
        assert result.error is None
        assert result.state.cash_cents == 500_000  # 20000 + 20000 - 35000
        assert result.state.holdings["A01"] == 550  # 400 - 200 + 350


class TestAC04InvalidOperations:
    """无效操作被拒绝、返回明确错误、不改变有效结果。"""

    def test_buy_a01_201_insufficient_cash(self, base_portfolio):
        state, instruments, _ = base_portfolio
        result = simulate_sequential(state, [("BUY", "A01", 201)], instruments)
        assert result.error is not None
        assert result.error.code == "INSUFFICIENT_CASH"
        assert result.error.op_index == 1
        assert result.applied == ()
        assert result.state == state  # 原状态不变

    def test_sell_a01_401_excess_quantity(self, base_portfolio):
        state, instruments, _ = base_portfolio
        result = simulate_sequential(state, [("SELL", "A01", 401)], instruments)
        assert result.error is not None
        assert result.error.code == "EXCESS_QUANTITY"
        assert result.state == state

    @pytest.mark.parametrize("quantity", [0, -5, 2.5, "10", None, True])
    def test_invalid_quantity_rejected(self, base_portfolio, quantity):
        state, instruments, _ = base_portfolio
        result = simulate_sequential(state, [("BUY", "A01", quantity)], instruments)
        assert result.error is not None
        assert result.error.code == "INVALID_QUANTITY"
        assert result.state == state

    def test_unknown_symbol_rejected(self, base_portfolio):
        state, instruments, _ = base_portfolio
        result = simulate_sequential(state, [("BUY", "ZZ9", 1)], instruments)
        assert result.error is not None
        assert result.error.code == "UNKNOWN_SYMBOL"
        assert result.state == state

    def test_integer_valued_float_accepted(self, base_portfolio):
        """数量为整数值的浮点（如 2.0）视为正整数接受，非整小数拒绝。"""
        state, instruments, _ = base_portfolio
        ok = simulate_sequential(state, [("BUY", "A03", 2.0)], instruments)
        assert ok.error is None
        assert ok.state.holdings["A03"] == 102

    def test_mid_list_failure_keeps_valid_prefix(self, base_portfolio):
        """中途失败：已生效操作保留，失败操作及其后操作不执行。"""
        state, instruments, _ = base_portfolio
        result = simulate_sequential(
            state, [("SELL", "A01", 100), ("BUY", "A01", 301), ("SELL", "A02", 1)], instruments
        )
        assert result.error is not None
        assert result.error.op_index == 2
        assert len(result.applied) == 1
        # 仅第一笔生效：现金 30,000，A01 持仓 300
        assert result.state.cash_cents == 3_000_000
        assert result.state.holdings["A01"] == 300
        assert result.state.holdings["A02"] == 400

    def test_base_state_not_mutated(self, base_portfolio):
        """原始组合在任何模拟后保持不变（隔离性）。"""
        state, instruments, _ = base_portfolio
        simulate_sequential(state, [("SELL", "A01", 100), ("BUY", "A02", 100)], instruments)
        simulate_sequential(state, [("BUY", "A01", 999)], instruments)
        assert state.cash_cents == 2_000_000
        assert state.holdings == {"A01": 400, "A02": 400, "A03": 100, "A04": 100, "A05": 30, "A06": 20}


class TestInvariantsAndEdges:
    def test_total_assets_invariant_under_valid_ops(self, base_portfolio):
        """固定价格、零费用下，模拟前后总资产不变。"""
        state, instruments, _ = base_portfolio
        result = simulate_sequential(
            state,
            [("SELL", "A01", 400), ("BUY", "A02", 800), ("SELL", "A06", 20), ("BUY", "A03", 5)],
            instruments,
        )
        assert result.error is None
        assert total_cents(result.state, instruments) == total_cents(state, instruments)

    def test_full_sell_out(self, base_portfolio):
        """清仓后该标的仍展示，市值/占比为 0.00，不触发提醒。"""
        state, instruments, rules = base_portfolio
        result = simulate_sequential(state, [("SELL", "A06", 20)], instruments)
        analysis = analyze(result.state, instruments, rules)
        a06 = next(i for i in analysis.items if i.symbol == "A06")
        assert a06.quantity == 0
        assert a06.market_value == "0.00"
        assert a06.weight_percent == "0.00"
        assert a06.single_triggered is False
        assert analysis.holdings_count == 5

    def test_rounding_edge_one_third(self):
        """1/3 占比展示为 33.33%，精确值大于 30% 阈值，触发。"""
        instruments = {
            "X": InstrumentDef("X", "标的X", "行业X", 10000),
            "Y": InstrumentDef("Y", "标的Y", "行业X", 10000),
        }
        state = PortfolioState("T", "测试", "CNY", "S", "2026-01-01T00:00:00+08:00", 0, {"X": 1, "Y": 2})
        rules = parse_portfolio({
            "portfolio_id": "T", "name": "测试", "currency": "CNY",
            "snapshot_id": "S", "snapshot_at": "2026-01-01T00:00:00+08:00",
            "cash": "0.00",
            "instruments": [
                {"symbol": "X", "name": "标的X", "industry": "行业X", "price": "100.00", "quantity": 1},
                {"symbol": "Y", "name": "标的Y", "industry": "行业X", "price": "100.00", "quantity": 2},
            ],
            "rules": {
                "single_instrument_threshold": "0.30", "industry_threshold": "0.50",
                "comparison": "strictly_greater_than", "cash_in_denominator": True,
                "cash_in_industry": False, "quantity": "positive_integer", "fees": "0.00",
            },
        })[2]
        analysis = analyze(state, instruments, rules)
        x = next(i for i in analysis.items if i.symbol == "X")
        assert x.weight == "0.33"
        assert x.weight_percent == "33.33"
        assert x.single_triggered is True  # 1/3 > 0.30 精确成立
        assert x.industry_weight_percent == "100.00"
        assert x.industry_triggered is True

    def test_cash_weight(self, base_portfolio):
        state, instruments, rules = base_portfolio
        analysis = analyze(state, instruments, rules)
        assert analysis.cash_weight == "0.20"
