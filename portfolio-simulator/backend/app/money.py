"""金额与占比的精度处理。

内部约定（详见 DESIGN.md「计算与异常」）：
- 金额一律以「分」为单位的整数（int）表示，例如 100.00 元 -> 10000 分；
- 占比使用 fractions.Fraction 精确有理数表示与比较，float 不参与任何计算；
- 阈值比较发生在展示舍入之前（任务书要求：strictly_greater_than）；
- 展示层统一 ROUND_HALF_UP 保留两位小数。
"""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from fractions import Fraction

_CENT = Decimal("0.01")


def parse_cents(raw: str) -> int:
    """将两位小数金额字符串解析为分（整数）。

    例如 "100.00" -> 10000，"20000.00" -> 2000000。
    精度超过分（如 "100.001"）或格式非法时报 ValueError。
    """
    if not isinstance(raw, str):
        raise ValueError(f"金额必须为字符串：{raw!r}")
    try:
        value = Decimal(raw.strip())
    except InvalidOperation as exc:
        raise ValueError(f"金额格式错误：{raw!r}") from exc
    if value < 0:
        raise ValueError(f"金额不能为负：{raw!r}")
    cents = value / _CENT
    if cents != cents.to_integral_value():
        raise ValueError(f"金额精度超过分：{raw!r}")
    return int(cents)


def format_cents(cents: int) -> str:
    """分 -> 两位小数字符串（ROUND_HALF_UP），例如 4000000 -> "40000.00"。"""
    return (Decimal(cents) / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP).__format__("f")


def parse_threshold(raw: str) -> Fraction:
    """将阈值字符串解析为精确 Fraction，例如 "0.30" -> 3/10。"""
    if not isinstance(raw, str):
        raise ValueError(f"阈值必须为字符串：{raw!r}")
    try:
        return Fraction(Decimal(raw.strip()))
    except InvalidOperation as exc:
        raise ValueError(f"阈值格式错误：{raw!r}") from exc


def weight_fraction(numerator_cents: int, denominator_cents: int) -> Fraction:
    """占比 = 分子金额（分） / 分母金额（分），精确有理数。"""
    if denominator_cents == 0:
        raise ValueError("占比分母不能为零")
    return Fraction(numerator_cents, denominator_cents)


def weight_gt(value: Fraction, threshold: Fraction) -> bool:
    """严格大于比较（等于不触发），使用舍入前的精确值。"""
    return value > threshold


def format_weight(fraction: Fraction) -> tuple[str, str]:
    """Fraction -> ("0.40", "40.00") 形式的 (权重, 百分比) 展示字符串。

    先量化到 4 位小数，再由同一小数派生权重与百分比，
    保证两者一致（不会出现 0.33 与 33.34 并存）。
    """
    precise = (
        Decimal(fraction.numerator) / Decimal(fraction.denominator)
    ).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
    weight = precise.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    percent = (precise * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return weight.__format__("f"), percent.__format__("f")
