"""money.py 精度与舍入单元测试。"""
from fractions import Fraction

import pytest

from app.money import (
    format_cents,
    format_weight,
    parse_cents,
    parse_threshold,
    weight_fraction,
    weight_gt,
)


class TestParseCents:
    def test_parse_two_decimals(self):
        assert parse_cents("100.00") == 10000
        assert parse_cents("20000.00") == 2000000
        assert parse_cents("0.00") == 0

    def test_parse_sub_cent_rejected(self):
        with pytest.raises(ValueError, match="精度超过分"):
            parse_cents("100.001")

    def test_parse_invalid_rejected(self):
        with pytest.raises(ValueError):
            parse_cents("abc")
        with pytest.raises(ValueError):
            parse_cents("-1.00")
        with pytest.raises(ValueError):
            parse_cents(100)


class TestFormatCents:
    def test_format(self):
        assert format_cents(4000000) == "40000.00"
        assert format_cents(5) == "0.05"
        assert format_cents(0) == "0.00"


class TestThreshold:
    def test_parse_threshold(self):
        assert parse_threshold("0.30") == Fraction(3, 10)
        assert parse_threshold("0.50") == Fraction(1, 2)

    def test_strictly_greater_equality_does_not_trigger(self):
        """任务书：等于阈值不触发。"""
        assert weight_gt(Fraction(3, 10), Fraction(3, 10)) is False
        assert weight_gt(Fraction(1, 2), Fraction(1, 2)) is False

    def test_strictly_greater(self):
        assert weight_gt(Fraction(31, 100), Fraction(3, 10)) is True

    def test_exact_comparison_before_rounding(self):
        """0.2999999 展示舍入后为 0.30，但精确值未达阈值，不触发。"""
        assert weight_gt(Fraction(2999999, 10000000), Fraction(3, 10)) is False
        assert weight_gt(Fraction(3000001, 10000000), Fraction(3, 10)) is True


class TestFormatWeight:
    def test_thirds(self):
        assert format_weight(Fraction(1, 3)) == ("0.33", "33.33")

    def test_forty_percent(self):
        assert format_weight(Fraction(2, 5)) == ("0.40", "40.00")

    def test_exact_half(self):
        assert format_weight(Fraction(1, 2)) == ("0.50", "50.00")

    def test_weight_fraction_zero_denominator_rejected(self):
        with pytest.raises(ValueError):
            weight_fraction(1, 0)
