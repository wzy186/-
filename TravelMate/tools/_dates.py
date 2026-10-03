"""日期标准化 — 让 Mock 数据的日期与真实当前日期保持一致。

用法：
    normalize_date("")           -> 明天 "2026-10-04"
    normalize_date("10月5号")    -> "2026-10-05"（自动补当前年份，过去日期顺延一年）
    normalize_date("明天")       -> "2026-10-04"
    normalize_date("2026/10/8")  -> "2026-10-08"
"""
from __future__ import annotations

import re
from datetime import date, timedelta

_WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}


def normalize_date(text: str, default_offset_days: int = 1) -> str:
    """把各种日期写法统一为 YYYY-MM-DD；解析失败返回空字符串。"""
    if not text:
        return (date.today() + timedelta(days=default_offset_days)).isoformat()
    text = text.strip()

    today = date.today()

    # 已是标准格式
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", text)
    if m:
        return _fmt(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})$", text)
    if m:
        return _fmt(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # 相对日期
    if "大后天" in text:
        return (today + timedelta(days=3)).isoformat()
    if "后天" in text:
        return (today + timedelta(days=2)).isoformat()
    if "明天" in text:
        return (today + timedelta(days=1)).isoformat()
    if "今天" in text or "当日" in text:
        return today.isoformat()
    m = re.search(r"(\d+)天[之后后]", text)
    if m:
        return (today + timedelta(days=int(m.group(1)))).isoformat()
    # 本周X（未来最近的一个）/ 下周X（下一个周一所在周）
    m = re.search(r"(下周|本周|[周星期])([一二三四五六日天])", text)
    if m:
        target = _WEEKDAY.get(m.group(2), 0)
        if m.group(1) == "下周":
            days_to_next_monday = (7 - today.weekday()) % 7 or 7
            return (today + timedelta(days=days_to_next_monday + target)).isoformat()
        days_ahead = (target - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return (today + timedelta(days=days_ahead)).isoformat()

    # 中文日期：10月5号 / 10月5日 / 10-5
    m = re.search(r"(\d{1,2})月(\d{1,2})[号日]?", text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        year = today.year
        try:
            d = date(year, month, day)
            if d < today:  # 已过去 → 明年
                d = date(year + 1, month, day)
            return d.isoformat()
        except ValueError:
            return ""
    m = re.match(r"^(\d{1,2})-(\d{1,2})$", text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        try:
            d = date(today.year, month, day)
            if d < today:
                d = date(today.year + 1, month, day)
            return d.isoformat()
        except ValueError:
            return ""

    return ""


def _fmt(y: int, mo: int, d: int) -> str:
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return ""
