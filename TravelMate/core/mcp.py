"""MCP-compatible tool protocol.

Each tool declares a JSON Schema describing its parameters.
This makes tools discoverable, validated, and reusable across frameworks."""

from __future__ import annotations
import json
from typing import Any
from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    """MCP-style tool schema declaration."""
    name: str
    description: str
    parameters: dict  # JSON Schema dict
    category: str = "query"  # query | action


class ToolResult(BaseModel):
    """Standardized tool execution result."""
    success: bool
    data: Any = None
    error: str = ""
    message: str = ""
    is_action: bool = False


# ── Tool Registry ──

_TOOL_SCHEMAS: dict[str, ToolSchema] = {}
_TOOL_EXECUTORS: dict[str, callable] = {}


def register_tool(schema: ToolSchema, executor: callable):
    """Register a tool with its schema and executor function."""
    _TOOL_SCHEMAS[schema.name] = schema
    _TOOL_EXECUTORS[schema.name] = executor


def get_tool_schema(name: str) -> ToolSchema | None:
    return _TOOL_SCHEMAS.get(name)


def get_all_tool_schemas() -> list[ToolSchema]:
    return list(_TOOL_SCHEMAS.values())


def execute_tool(name: str, args: dict) -> ToolResult:
    """Execute a tool by name with validated args."""
    schema = _TOOL_SCHEMAS.get(name)
    if not schema:
        return ToolResult(success=False, error=f"Unknown tool: {name}")

    executor = _TOOL_EXECUTORS.get(name)
    if not executor:
        return ToolResult(success=False, error=f"No executor for: {name}")

    try:
        result_str, is_action = executor(name, args)
        try:
            data = json.loads(result_str)
        except json.JSONDecodeError:
            data = {"raw": result_str}

        return ToolResult(
            success=data.get("success", True) if isinstance(data, dict) else True,
            data=data,
            message=data.get("message", "") if isinstance(data, dict) else "",
            is_action=is_action,
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def get_tools_for_prompt() -> str:
    """Generate tool descriptions for the system prompt."""
    lines = []
    for schema in _TOOL_SCHEMAS.values():
        cat = "操作" if schema.category == "action" else "查询"
        params = ", ".join(f'"{k}": {v.get("description","")}' for k, v in schema.parameters.get("properties", {}).items())
        lines.append(f"- {schema.name}: {schema.description} [{cat}] {{{params}}}")
    return "\n".join(lines)


# ── Register all tools ──

def _init_registry():
    """Register all project tools into the MCP registry."""
    from tools.weather import WeatherTool
    from tools.flight import FlightTool
    from tools.exchange import ExchangeTool
    from tools.hotel import HotelTool
    from tools.amap import AmapTool
    from tools.budget_tool import BudgetTool
    from tools.attraction import AttractionTool
    from tools.translate import TranslateTool

    tool_instances = {
        "weather": WeatherTool(),
        "flight": FlightTool(),
        "exchange": ExchangeTool(),
        "hotel": HotelTool(),
        "budget": BudgetTool(),
        "attraction": AttractionTool(),
        "translate": TranslateTool(),
        "route": AmapTool("route"),
        "nearby": AmapTool("nearby"),
        "geocode": AmapTool("geocode"),
        "district": AmapTool("district"),
    }

    # Query tools
    query_schemas = {
        "weather": ToolSchema(name="weather", description="查询目的地天气（7日预报+穿衣建议）", category="query",
            parameters={"type": "object", "properties": {"destination": {"type": "string", "description": "城市名"}, "days": {"type": "integer", "description": "预报天数"}}}),
        "flight": ToolSchema(name="flight", description="查询航班信息（多航司比价）", category="query",
            parameters={"type": "object", "properties": {"departure": {"type": "string", "description": "出发城市"}, "destination": {"type": "string", "description": "到达城市"}, "date": {"type": "string", "description": "日期"}}}),
        "exchange": ToolSchema(name="exchange", description="汇率换算", category="query",
            parameters={"type": "object", "properties": {"amount": {"type": "number", "description": "金额"}, "from": {"type": "string", "description": "源货币"}, "to": {"type": "string", "description": "目标货币"}}}),
        "hotel": ToolSchema(name="hotel", description="酒店推荐", category="query",
            parameters={"type": "object", "properties": {"destination": {"type": "string", "description": "城市"}, "budget_per_night": {"type": "number", "description": "每晚预算"}, "style": {"type": "string", "description": "风格"}}}),
        "attraction": ToolSchema(name="attraction", description="景点推荐（RAG增强）", category="query",
            parameters={"type": "object", "properties": {"city": {"type": "string", "description": "城市"}, "query": {"type": "string", "description": "搜索词"}}}),
        "budget": ToolSchema(name="budget", description="预算计算与分配", category="query",
            parameters={"type": "object", "properties": {"budget": {"type": "number", "description": "总预算"}, "days": {"type": "integer", "description": "天数"}, "destination": {"type": "string", "description": "目的地"}, "style": {"type": "string", "description": "风格"}}}),
        "translate": ToolSchema(name="translate", description="多语言翻译", category="query",
            parameters={"type": "object", "properties": {"text": {"type": "string", "description": "原文"}, "target": {"type": "string", "description": "目标语言代码"}}}),
        "route": ToolSchema(name="route", description="路线规划（高德地图）", category="query",
            parameters={"type": "object", "properties": {"origin_name": {"type": "string", "description": "起点"}, "destination_name": {"type": "string", "description": "终点"}, "mode": {"type": "string", "description": "方式"}}}),
        "nearby": ToolSchema(name="nearby", description="周边搜索", category="query",
            parameters={"type": "object", "properties": {"location_name": {"type": "string", "description": "位置"}, "keywords": {"type": "string", "description": "关键词"}, "radius": {"type": "integer", "description": "半径(米)"}}}),
        "geocode": ToolSchema(name="geocode", description="地理编码", category="query",
            parameters={"type": "object", "properties": {"address": {"type": "string", "description": "地址"}}}),
        "district": ToolSchema(name="district", description="行政区划查询", category="query",
            parameters={"type": "object", "properties": {"keywords": {"type": "string", "description": "查询词"}}}),
    }

    # Action tools
    action_schemas = {
        "book_flight": ToolSchema(name="book_flight", description="预订航班", category="action",
            parameters={"type": "object", "properties": {"airline": {"type": "string", "description": "航班号"}, "departure": {"type": "string", "description": "出发"}, "arrival": {"type": "string", "description": "到达"}, "date": {"type": "string", "description": "日期"}, "passenger": {"type": "string", "description": "乘客"}, "price": {"type": "number", "description": "价格"}, "seat": {"type": "string", "description": "座位偏好"}, "meal": {"type": "string", "description": "餐食"}}}),
        "book_hotel": ToolSchema(name="book_hotel", description="预订酒店", category="action",
            parameters={"type": "object", "properties": {"name": {"type": "string", "description": "酒店名"}, "city": {"type": "string", "description": "城市"}, "check_in": {"type": "string", "description": "入住日期"}, "check_out": {"type": "string", "description": "退房日期"}, "guest": {"type": "string", "description": "入住人"}, "room_type": {"type": "string", "description": "房型"}, "price_per_night": {"type": "number", "description": "每晚价格"}, "nights": {"type": "integer", "description": "晚数"}, "guests": {"type": "integer", "description": "人数"}}}),
        "add_spot": ToolSchema(name="add_spot", description="添加景点到行程", category="action",
            parameters={"type": "object", "properties": {"name": {"type": "string", "description": "景点名"}, "city": {"type": "string", "description": "城市"}, "note": {"type": "string", "description": "备注"}}}),
        "save_phrase": ToolSchema(name="save_phrase", description="收藏翻译短语", category="action",
            parameters={"type": "object", "properties": {"zh": {"type": "string", "description": "中文"}, "foreign": {"type": "string", "description": "外语"}, "pron": {"type": "string", "description": "发音"}, "lang": {"type": "string", "description": "语言"}}}),
        "add_reminder": ToolSchema(name="add_reminder", description="添加提醒", category="action",
            parameters={"type": "object", "properties": {"text": {"type": "string", "description": "提醒内容"}, "date": {"type": "string", "description": "日期"}, "type": {"type": "string", "description": "提醒类型"}}}),
        "set_note": ToolSchema(name="set_note", description="添加行程备注", category="action",
            parameters={"type": "object", "properties": {"key": {"type": "string", "description": "备注key"}, "content": {"type": "string", "description": "备注内容"}}}),
    }

    # Unified executor for query tools
    def _query_executor(name, args):
        tool = tool_instances.get(name)
        if tool:
            return tool.run(args), False
        return json.dumps({"error": f"Unknown query tool: {name}"}, ensure_ascii=False), False

    # Unified executor for action tools
    def _action_executor(name, args):
        from utils.storage import (
            add_booking, add_itinerary_spot, save_phrase,
            add_reminder, save_note, get_current_user,
        )
        username = get_current_user() or "default"

        if name == "book_flight":
            booking_id = add_booking({
                "type": "flight", "name": f"{args.get('airline','')} {args.get('departure','')}→{args.get('arrival','')}",
                "airline": args.get("airline", ""), "route": f"{args.get('departure','')}→{args.get('arrival','')}",
                "date": args.get("date", "待确认"), "depart": args.get("depart", ""), "arrive": args.get("arrive", ""),
                "price": args.get("price", 0), "passenger": args.get("passenger", ""),
                "seat": args.get("seat", "无偏好"), "meal": args.get("meal", "标准"), "status": "已确认",
            }, username)
            return json.dumps({"success": True, "booking_id": booking_id,
                "message": f"✅ 航班预订成功！订单号: {booking_id}\n{args.get('airline','')} {args.get('departure','')}→{args.get('arrival','')} | {args.get('date','')} | ¥{args.get('price',0):,} | 乘客: {args.get('passenger','')}"
            }, ensure_ascii=False), True

        if name == "book_hotel":
            nights = args.get("nights", 1)
            total = args.get("price_per_night", 0) * nights
            booking_id = add_booking({
                "type": "hotel", "name": args.get("name", ""), "city": args.get("city", ""),
                "room_type": args.get("room_type", "标准间"), "check_in": args.get("check_in", "待确认"),
                "check_out": args.get("check_out", "待确认"), "nights": nights,
                "price_per_night": args.get("price_per_night", 0), "total": total,
                "guest": args.get("guest", ""), "guests": args.get("guests", 1), "status": "已确认",
            }, username)
            return json.dumps({"success": True, "booking_id": booking_id,
                "message": f"✅ 酒店预订成功！订单号: {booking_id}\n{args.get('name','')} | {args.get('room_type','')} | {nights}晚 | ¥{total:,}"
            }, ensure_ascii=False), True

        if name == "add_spot":
            add_itinerary_spot({"name": args.get("name", ""), "city": args.get("city", ""), "note": args.get("note", "")}, username)
            return json.dumps({"success": True, "message": f"✅ 已将「{args.get('name','')}」添加到行程"}, ensure_ascii=False), True

        if name == "save_phrase":
            save_phrase({"zh": args.get("zh", ""), "foreign": args.get("foreign", ""), "pron": args.get("pron", ""), "lang": args.get("lang", "")}, username)
            return json.dumps({"success": True, "message": f"✅ 已收藏翻译短语: {args.get('zh','')} → {args.get('foreign','')}"}, ensure_ascii=False), True

        if name == "add_reminder":
            add_reminder({"text": args.get("text", ""), "date": args.get("date", ""), "type": args.get("type", "旅行提醒")}, username)
            return json.dumps({"success": True, "message": f"✅ 已添加提醒: {args.get('text','')} ({args.get('date','')})"}, ensure_ascii=False), True

        if name == "set_note":
            save_note(args.get("key", "note"), args.get("content", ""), username)
            return json.dumps({"success": True, "message": "✅ 已保存备注"}, ensure_ascii=False), True

        return json.dumps({"error": f"Unknown action tool: {name}"}, ensure_ascii=False), False

    # Register all
    for name, schema in query_schemas.items():
        register_tool(schema, _query_executor)
    for name, schema in action_schemas.items():
        register_tool(schema, _action_executor)


# Auto-initialize on import
_init_registry()
