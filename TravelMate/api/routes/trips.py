from fastapi import APIRouter, HTTPException
from models.schemas import (
    TripRequest, BudgetRequest, ExpenseRequest, NearbyRequest,
    RouteRequest, TranslateRequest, SettleRequest, VisaRequest,
)
from core.llm import chat, chat_json
from core.prompts import *
from core.rag import get_context, search, index_destinations
from core.memory import add_trip_history, get_trip_history
from tools.weather import WeatherTool
from tools.flight import FlightTool
from tools.exchange import ExchangeTool
from tools.translate import TranslateTool
from tools.attraction import AttractionTool
from tools.budget_tool import BudgetTool
from tools.hotel import HotelTool
from tools.amap import AmapTool
from models.database import SessionLocal, Trip, Expense, SavedItinerary, init_db
import json
import uuid

router = APIRouter()


@router.post("/trips/plan")
def plan_trip(req: TripRequest):
    intent = {"穷游版": "plan_budget", "豪华版": "plan_luxury"}.get(getattr(req, 'plan_type', '舒适版'), "plan")
    request_str = f"去{req.destination}{req.days}天，预算{req.budget}元，{req.style}，{req.season}季出行"
    context = get_context(req.destination)
    prompt = PLAN_PROMPT.format(request=request_str, context=context)
    result = chat_json(prompt, intent=intent)
    if not result:
        try:
            result = json.loads(chat("", intent=intent))
        except Exception:
            result = {}

    # Save to DB
    try:
        init_db()
        db = SessionLocal()
        trip_id = str(uuid.uuid4())[:8]
        db_trip = Trip(id=trip_id, destination=req.destination, days=req.days,
                       budget=req.budget, style=req.style, plan_json=json.dumps(result, ensure_ascii=False))
        db.add(db_trip)
        db.commit()
        db.close()
        result["trip_id"] = trip_id
    except Exception:
        pass

    add_trip_history({"destination": req.destination, "days": req.days, "budget": req.budget})
    return result


@router.get("/trips")
def list_trips():
    try:
        init_db()
        db = SessionLocal()
        trips = db.query(Trip).order_by(Trip.created_at.desc()).limit(20).all()
        db.close()
        return [{"id": t.id, "destination": t.destination, "days": t.days, "budget": t.budget, "style": t.style, "status": t.status, "created_at": str(t.created_at)} for t in trips]
    except Exception:
        return []


@router.get("/trips/{trip_id}")
def get_trip(trip_id: str):
    try:
        init_db()
        db = SessionLocal()
        trip = db.query(Trip).filter(Trip.id == trip_id).first()
        db.close()
        if trip:
            return {"id": trip.id, "destination": trip.destination, "days": trip.days, "budget": trip.budget, "style": trip.style, "plan": json.loads(trip.plan_json) if trip.plan_json else {}, "status": trip.status}
        raise HTTPException(status_code=404, detail="Trip not found")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Database error")


@router.post("/budget/calculate")
def calculate_budget(req: BudgetRequest):
    tool = BudgetTool()
    result = tool.run({"budget": req.budget, "days": req.days, "destination": req.destination, "currency": req.currency})
    try:
        return json.loads(result)
    except Exception:
        return {"error": "Budget calculation failed"}


@router.post("/expenses")
def add_expense(req: ExpenseRequest):
    try:
        init_db()
        db = SessionLocal()
        exp = Expense(id=str(uuid.uuid4())[:8], trip_id=req.trip_id, payer=req.payer,
                      item=req.item, amount=req.amount, currency=req.currency, category=req.category)
        db.add(exp)
        db.commit()
        db.close()
        return {"status": "ok", "expense_id": exp.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/expenses/{trip_id}")
def get_expenses(trip_id: str):
    try:
        init_db()
        db = SessionLocal()
        expenses = db.query(Expense).filter(Expense.trip_id == trip_id).all()
        db.close()
        return [{"id": e.id, "payer": e.payer, "item": e.item, "amount": e.amount, "currency": e.currency, "category": e.category} for e in expenses]
    except Exception:
        return []


@router.post("/weather")
def query_weather(destination: str = "东京", days: int = 7):
    tool = WeatherTool()
    result = tool.run({"destination": destination, "days": days})
    try:
        return json.loads(result)
    except Exception:
        return {"error": "Weather query failed"}


@router.post("/flights")
def query_flights(departure: str = "北京", destination: str = "东京", date: str = ""):
    tool = FlightTool()
    result = tool.run({"departure": departure, "destination": destination, "date": date})
    try:
        return json.loads(result)
    except Exception:
        return {"error": "Flight query failed"}


@router.post("/hotels")
def query_hotels(destination: str = "东京", budget_per_night: int = 800, style: str = "舒适"):
    tool = HotelTool()
    result = tool.run({"destination": destination, "budget_per_night": budget_per_night, "style": style})
    try:
        return json.loads(result)
    except Exception:
        return {"error": "Hotel query failed"}


@router.post("/exchange")
def exchange_currency(amount: float = 1000, from_curr: str = "CNY", to_curr: str = "JPY"):
    tool = ExchangeTool()
    result = tool.run({"amount": amount, "from": from_curr, "to": to_curr})
    try:
        return json.loads(result)
    except Exception:
        return {"error": "Exchange calculation failed"}


@router.post("/translate")
def translate_text(req: TranslateRequest):
    tool = TranslateTool()
    result = tool.run({"text": req.text, "target": req.target, "scene": req.scene})
    try:
        return json.loads(result)
    except Exception:
        return {"error": "Translation failed"}


@router.post("/nearby")
def search_nearby(req: NearbyRequest):
    tool = AmapTool("nearby")
    result = tool.run({"location_name": req.location, "keywords": req.keywords, "radius": req.radius})
    try:
        return json.loads(result)
    except Exception:
        return {"error": "Nearby search failed"}


@router.post("/route")
def plan_route(req: RouteRequest):
    tool = AmapTool("route")
    result = tool.run({"origin_name": req.origin, "destination_name": req.destination, "mode": req.mode})
    try:
        return json.loads(result)
    except Exception:
        return {"error": "Route planning failed"}


@router.post("/settle")
def calculate_settle(req: SettleRequest):
    result = chat_json(SETTLE_PROMPT.format(expenses=json.dumps(req.expenses, ensure_ascii=False), members=json.dumps(req.members, ensure_ascii=False)), intent="settle")
    if not result:
        try:
            result = json.loads(chat("", intent="settle"))
        except Exception:
            result = {}
    return result


@router.post("/visa")
def query_visa(req: VisaRequest):
    result = chat_json(VISA_PROMPT.format(destination=req.destination, departure=req.departure), intent="visa")
    if not result:
        try:
            result = json.loads(chat("", intent="visa"))
        except Exception:
            result = {}
    return result


@router.post("/guide")
def generate_guide(destination: str = "东京"):
    context = get_context(destination)
    result = chat(GUIDE_PROMPT.format(destination=destination, context=context), intent="guide")
    if not result or len(result) < 50:
        result = chat("", intent="guide")
    return {"destination": destination, "guide": result}


@router.post("/packing")
def generate_packing(destination: str = "东京", season: str = "夏", days: int = 5, activities: str = "温泉,摄影"):
    result = chat_json(PACKING_PROMPT.format(destination=destination, season=season, days=days, activities=activities), intent="packing")
    if not result:
        try:
            result = json.loads(chat("", intent="packing"))
        except Exception:
            result = {}
    return result


@router.get("/knowledge/search")
def search_knowledge(q: str = "东京 景点", top_k: int = 5):
    results = search(q, top_k)
    return {"query": q, "results": results, "count": len(results)}


@router.post("/knowledge/index")
def rebuild_index():
    count = index_destinations()
    return {"status": "ok", "chunks": count}


@router.post("/attractions")
def query_attractions(city: str = "东京"):
    tool = AttractionTool()
    result = tool.run({"city": city, "query": f"{city} 景点"})
    try:
        return json.loads(result)
    except Exception:
        return {"error": "Attraction query failed"}


@router.post("/sos")
def get_sos_info(location: str = "东京", country: str = "日本"):
    result = chat_json(SOS_PROMPT.format(location=location, country=country), intent="sos")
    if not result:
        try:
            result = json.loads(chat("", intent="sos"))
        except Exception:
            result = {}
    return result
