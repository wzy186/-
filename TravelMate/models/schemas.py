from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class TripRequest(BaseModel):
    destination: str
    days: int = 5
    budget: float = 15000
    style: str = "文化体验"
    season: str = "夏"
    plan_type: str = "舒适版"


class BudgetRequest(BaseModel):
    destination: str
    budget: float
    days: int
    currency: str = "CNY"


class ExpenseRequest(BaseModel):
    trip_id: str
    payer: str
    item: str
    amount: float
    currency: str = "CNY"
    category: str = "other"


class NearbyRequest(BaseModel):
    location: str
    keywords: str = "餐厅"
    radius: int = 3000


class RouteRequest(BaseModel):
    origin: str
    destination: str
    mode: str = "驾车"


class TranslateRequest(BaseModel):
    text: str = ""
    target: str = "ja"
    scene: str = ""


class SettleRequest(BaseModel):
    members: list[str]
    expenses: list[dict]


class VisaRequest(BaseModel):
    destination: str
    departure: str = "中国"


class GuideRequest(BaseModel):
    destination: str


class PackingRequest(BaseModel):
    destination: str = "东京"
    season: str = "夏"
    days: int = 5
    activities: str = "温泉,摄影"


class SosRequest(BaseModel):
    location: str = "东京"
    country: str = "日本"
