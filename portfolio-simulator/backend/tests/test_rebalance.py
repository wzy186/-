"""目标权重调仓建议引擎测试（期望值均经人工笔算复核）。"""
from fractions import Fraction

from app.engine import PortfolioState, InstrumentDef, analyze, round_half_up, suggest_rebalance


def total_cents(state: PortfolioState, instruments: dict[str, InstrumentDef]) -> int:
    return state.cash_cents + sum(
        d.price_cents * state.holdings.get(s, 0) for s, d in instruments.items()
    )


class TestRoundHalfUp:
    def test_half_up(self):
        assert round_half_up(Fraction(1, 2)) == 1
        assert round_half_up(Fraction(3, 2)) == 2
        assert round_half_up(Fraction(333, 2)) == 167  # 166.5 -> 167
        assert round_half_up(Fraction(1, 3)) == 0


class TestEqualWeight:
    """等权分散：6 标的各 1/6、现金目标 0。

    手算：目标市值 1,666,666.67 分 → 目标数量 A01/A03/A05/A06=167、A02/A04=333。
    先卖后买：卖 A01×233、A02×67（现金 46,650）；买 A03×67、A04×233、A05×137、
    A06×147 共 46,750，最后一笔现金不足 100 元 → 自动减量 A06→146，现金恰好为 0。
    """

    def test_equal_weight_suggestion(self, base_portfolio):
        state, instruments, rules = base_portfolio
        targets = {symbol: Fraction(1, len(instruments)) for symbol in instruments}
        ops, result = suggest_rebalance(state, targets, instruments)

        assert result.error is None
        assert ops == [
            ("SELL", "A01", 233),
            ("SELL", "A02", 67),
            ("BUY", "A03", 67),
            ("BUY", "A04", 233),
            ("BUY", "A05", 137),
            ("BUY", "A06", 146),  # 147 -> 146（现金不足自动减量）
        ]
        assert result.state.cash_cents == 0
        assert result.state.holdings == {
            "A01": 167, "A02": 333, "A03": 167, "A04": 333, "A05": 167, "A06": 166,
        }
        assert total_cents(result.state, instruments) == 10_000_000  # 总资产不变

    def test_equal_weight_clears_concentration(self, base_portfolio):
        """等权分散后最大占比 16.70%，集中度超限全部解除。"""
        state, instruments, rules = base_portfolio
        targets = {symbol: Fraction(1, len(instruments)) for symbol in instruments}
        _, result = suggest_rebalance(state, targets, instruments)
        analysis = analyze(result.state, instruments, rules)
        assert analysis.single_concentration == ()
        assert analysis.industry_concentration == ()
        assert max(float(i.weight_percent) for i in analysis.items) <= 17.0

    def test_original_untouched(self, base_portfolio):
        state, instruments, _ = base_portfolio
        targets = {symbol: Fraction(1, len(instruments)) for symbol in instruments}
        suggest_rebalance(state, targets, instruments)
        assert state.holdings == {"A01": 400, "A02": 400, "A03": 100, "A04": 100, "A05": 30, "A06": 20}
        assert state.cash_cents == 2_000_000


class TestTargetWeights:
    def test_partial_targets_keep_others(self, base_portfolio):
        """A01→25%、A02→25%，其余保持：卖 A01×150、买 A02×100，现金 30,000。"""
        state, instruments, rules = base_portfolio
        targets = {"A01": Fraction(1, 4), "A02": Fraction(1, 4)}
        ops, result = suggest_rebalance(state, targets, instruments)
        assert ops == [("SELL", "A01", 150), ("BUY", "A02", 100)]
        assert result.error is None
        assert result.state.cash_cents == 3_000_000
        assert result.state.holdings["A01"] == 250
        assert result.state.holdings["A02"] == 500
        assert result.state.holdings["A03"] == 100  # 未指定保持
        analysis = analyze(result.state, instruments, rules)
        a01 = next(i for i in analysis.items if i.symbol == "A01")
        assert a01.weight_percent == "25.00"
        assert a01.single_triggered is False

    def test_aggressive_target_reduced_to_feasible(self, base_portfolio):
        """A01→90% 但现金不足：建议被逐股减量至可行（买 500 → 200），现金为 0。"""
        state, instruments, _ = base_portfolio
        ops, result = suggest_rebalance(state, {"A01": Fraction(9, 10)}, instruments)
        assert result.error is None
        assert ops == [("BUY", "A01", 200)]
        assert result.state.cash_cents == 0
        assert result.state.holdings["A01"] == 600

    def test_targets_equal_current_is_noop(self, base_portfolio):
        """目标权重等于当前占比：无任何操作，无错误。"""
        state, instruments, _ = base_portfolio
        targets = {
            "A01": Fraction(2, 5), "A02": Fraction(1, 5), "A03": Fraction(1, 10),
            "A04": Fraction(1, 20), "A05": Fraction(3, 100), "A06": Fraction(1, 50),
        }
        ops, result = suggest_rebalance(state, targets, instruments)
        assert ops == []
        assert result.error is None
        assert result.state == state
