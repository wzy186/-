"""API 入参模型。

注意：quantity 声明为 int | float，使小数数量（如 2.5）能到达引擎，
由引擎统一以 INVALID_QUANTITY 业务错误拒绝；整数值浮点（2.0）按正整数接受。
"""
from typing import Literal

from pydantic import BaseModel


class OpIn(BaseModel):
    action: Literal["BUY", "SELL"]
    symbol: str
    quantity: int | float


class SimulateRequest(BaseModel):
    operations: list[OpIn] = []


class PlanCreate(BaseModel):
    name: str
    operations: list[OpIn] = []


class RebalanceRequest(BaseModel):
    """目标权重调仓建议：equal=等权分散；targets=自定义权重（symbol -> "0.25"，
    未列出的标的保持当前持仓，权重之和 ≤ 1，现金占比为剩余部分）。"""

    mode: Literal["equal", "targets"] = "targets"
    targets: dict[str, str] = {}
