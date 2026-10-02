"""Supervisor 调度 Agent — 意图识别与任务派发。

工作方式：
- 有 LLM：让 LLM 输出 JSON 任务单 {"tasks":[{"agent","instruction"}]}（最多 3 个，按执行顺序）
- 无 LLM（Mock 模式）：关键词规则路由，支持识别复合意图拆成多任务
- 任务写入 state.tasks，顶层编排器据此路由到对应专家子图
"""

from __future__ import annotations

import json

from core.graph import AgentState
from core.llm import chat_json, is_llm_available

SUPERVISOR_PROMPT = """你是 TravelMate 多 Agent 系统的调度中心（Supervisor）。你的唯一职责是理解用户请求，
把它拆解、指派给最合适的专家 Agent，不负责回答内容本身。

## 可调度的专家 Agent
- route_agent: 路线规划/导航/地图/周边搜索/地理编码/行政区划（高德地图）
- ticket_agent: 火车票/高铁/12306查询与预订、机票查询与预订
- travel_agent: 行程规划/酒店/天气/景点/预算/汇率/翻译/清单/提醒/收藏
- qa_agent: 目的地知识问答（签证/文化/安全/SOS/攻略咨询），无工具的纯知识问题

## 输出格式（严格 JSON，不要输出其他内容）
{"tasks": [{"agent": "<agent名>", "instruction": "<改写后、自包含的子任务指令>"}]}

## 规则
1. 最多拆 3 个任务，按合理执行顺序排列（先查询后操作，先行程后预订）
2. instruction 必须自包含：补全目的地/日期/人数等从上下文推断的信息
3. 与旅行无关的闲聊/常识问题 → qa_agent
4. 单一简单请求只给 1 个任务
"""


def _route_by_keywords(text: str) -> list[dict]:
    """Mock 模式的关键词路由（无 LLM 兜底）。"""
    tasks: list[dict] = []

    route_kw = ["路线", "怎么走", "怎么去", "导航", "地图", "附近", "周边", "多远",
                "驾车", "公交", "地铁路线", "步行去", "骑行", " gps", "导航去"]
    ticket_kw = ["火车", "高铁", "动车", "12306", "车票", "余票", "火车票", "订火车",
                 "机票", "航班", "订机票", "买机票", "订票", "买票"]
    travel_kw = ["规划", "行程", "酒店", "天气", "汇率", "翻译", "预算", "清单",
                 "景点", "攻略", "签证", "日记", "提醒", "收藏", "加入"]

    is_route = any(k in text for k in route_kw)
    is_ticket = any(k in text for k in ticket_kw)
    is_travel = any(k in text for k in travel_kw)

    if is_ticket:
        tasks.append({"agent": "ticket_agent", "instruction": text})
    if is_route:
        tasks.append({"agent": "route_agent", "instruction": text})
    if is_travel and not (is_route or is_ticket):
        tasks.append({"agent": "travel_agent", "instruction": text})

    if not tasks:
        tasks.append({"agent": "qa_agent", "instruction": text})
    return tasks[:3]


def node_supervisor(state: AgentState) -> AgentState:
    """调度节点：产出任务队列 state.tasks。"""
    if state.metadata.get("guardrail_blocked"):
        return state

    # 多任务回路：队列里还有剩余任务，直接放行给下一个子 Agent
    if state.tasks:
        state.thinking.append(f"[调度Agent] 继续派发剩余任务 → {state.tasks[0]['agent']}")
        return state

    text = state.user_input or ""
    tasks: list[dict] = []

    if is_llm_available():
        prompt = f"用户请求：{text}"
        profile_hint = state.metadata.get("profile_hint", "")
        if profile_hint:
            prompt += f"\n（用户画像：{profile_hint.strip()}）"
        result = chat_json(prompt, system=SUPERVISOR_PROMPT, intent="")
        raw_tasks = result.get("tasks") or []
        for t in raw_tasks[:3]:
            if isinstance(t, dict) and t.get("agent") in ("route_agent", "ticket_agent", "travel_agent", "qa_agent"):
                tasks.append({
                    "agent": t["agent"],
                    "instruction": (t.get("instruction") or text).strip(),
                })
        if not tasks:
            tasks = _route_by_keywords(text)
    else:
        tasks = _route_by_keywords(text)

    state.tasks = tasks
    agents = " → ".join(t["agent"] for t in tasks)
    state.thinking.append(f"[调度Agent] 识别意图，派发任务：{agents}")
    state.metadata["supervisor_tasks"] = json.dumps(tasks, ensure_ascii=False)
    return state


def router_supervisor(state: AgentState) -> str:
    """顶层条件路由：返回下一个子 Agent 节点名。"""
    if state.tasks:
        return state.tasks[0]["agent"]
    return "qa_agent"


def pop_task(state: AgentState) -> dict:
    """弹出队首任务，由各子 Agent 包装节点在进入前调用。"""
    if state.tasks:
        return state.tasks.pop(0)
    return {}


def route_after_specialist(state: AgentState) -> str:
    """子 Agent 完成后的路由：中断 → 结束等待确认；还有任务 → 回调度；否则收尾。"""
    if state.interrupted:
        return "__end__"
    if state.tasks:
        return "supervisor"
    return "guardrail_output"
