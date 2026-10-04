"""Supervisor 调度 Agent — Plan-and-Execute 动态多 Agent 调度。

工作方式：
- 有 LLM：让 LLM 输出分阶段执行计划 {"stages": [[任务,...], ...]}，
  同一阶段内的任务**无依赖、并行执行**（引擎 fan-out），阶段之间顺序执行
- 无 LLM（Mock 模式）：关键词规则生成单阶段并行计划
- Agent 间移交（handoff）：子 Agent 执行中发现超出职责的子请求，
  可通过 [call:handoff] 动态追加执行阶段，由调度器接续派发
"""

from __future__ import annotations

import json
import re

from core.graph import AgentState
from core.llm import chat_json, is_llm_available

VALID_AGENTS = ("route_agent", "ticket_agent", "travel_agent", "qa_agent")

SUPERVISOR_PROMPT = """你是 TravelMate 多 Agent 系统的调度中心（Supervisor / Planner）。
你的职责：把用户请求规划成可执行的**分阶段计划**，不负责回答内容本身。

## 可调度的专家 Agent
- route_agent: 路线规划/导航/地图/周边搜索/地理编码/行政区划（高德）
- ticket_agent: 火车票/高铁/12306查询与预订、机票查询与预订
- travel_agent: 行程规划/酒店/天气/景点/预算/汇率/翻译/清单/提醒/收藏
- qa_agent: 目的地知识问答（签证/文化/安全/SOS/攻略咨询），无工具的纯知识问题

## 输出格式（严格 JSON，不要输出其他内容）
{"stages": [
  [{"agent": "<agent名>", "instruction": "<自包含子任务>"}, ...],   ← 阶段1
  [{"agent": "<agent名>", "instruction": "<自包含子任务>"}, ...]    ← 阶段2（依赖阶段1的结果）
]}

## 规划规则
1. **同一阶段内的任务必须相互无依赖**（可并行执行）；有依赖关系的任务必须放到后面的阶段
   例：先查天气再规划行程 → 阶段1=[查天气]，阶段2=[生成行程]
   例：查火车票+查路线（互不依赖）→ 阶段1=[查票, 查路线]（并行）
2. 最多 3 个阶段，每阶段最多 2 个任务
3. instruction 必须自包含：补全目的地/日期/人数等从上下文推断的信息
4. 与旅行无关的闲聊/常识问题 → 单阶段单任务 qa_agent
5. 简单请求只给 1 个阶段 1 个任务，不要过度拆分

## 硬性规则（必须遵守）
- **凡是需要实时/动态数据的请求，绝对禁止派给 qa_agent**——它没有工具：
  天气（含对比/适合出行吗）、余票/票价、路线/距离、价格、汇率 → 派给对应工具型 Agent
- 天气查询/对比/穿衣建议 → travel_agent；"A和B哪个天气好"这类对比也是 travel_agent
- 示例：
  - "北京和东京哪边现在更热" → travel_agent（instruction: 查询北京和东京当前天气并对比温度）
  - "明天下雨吗" → travel_agent
  - "从北京站到北京西站怎么过去" → route_agent（instruction: 规划从北京站到北京西站的路线）
  - "A到B怎么过去/怎么到/如何前往" → route_agent
  - "日本签证需要什么材料" → qa_agent（纯知识，无需实时数据）
"""


_REALTIME_KW = ["天气", "气温", "温度", "下雨", "降雨", "热不热", "冷不冷", "余票", "票价",
                "路线", "怎么走", "导航", "汇率", "多少公里", "多远", "价格", "比价",
                "地铁", "公交", "驾车", "打车", "步行去"]
# 交通方式问法正则：怎么过去/怎么到/如何前往/坐地铁 等
_ROUTE_PAT = re.compile(r"怎么(过去|走|去|到|前往|到达)|如何(去|前往|到达)|(坐|乘|搭).{0,6}(地铁|公交|车)|最近的路")


def _has_realtime_intent(text: str) -> bool:
    return any(k in text for k in _REALTIME_KW) or bool(_ROUTE_PAT.search(text))


def _plan_by_keywords(text: str) -> list[list[dict]]:
    """Mock 模式的关键词规划（无 LLM 兜底）：命中的意图并行执行。"""
    route_kw = ["路线", "怎么走", "怎么去", "导航", "地图", "附近", "周边", "多远",
                "驾车", "公交", "地铁路线", "步行去", "骑行", "导航去"]
    ticket_kw = ["火车", "高铁", "动车", "12306", "车票", "余票", "火车票", "订火车",
                 "机票", "航班", "订机票", "买机票", "订票", "买票"]
    travel_kw = ["规划", "行程", "酒店", "天气", "汇率", "翻译", "预算", "清单",
                 "景点", "攻略", "签证", "日记", "提醒", "收藏", "加入"]

    stage: list[dict] = []
    if any(k in text for k in ticket_kw):
        stage.append({"agent": "ticket_agent", "instruction": text})
    if any(k in text for k in route_kw) or _ROUTE_PAT.search(text):
        stage.append({"agent": "route_agent", "instruction": text})
    if not stage and any(k in text for k in travel_kw):
        stage.append({"agent": "travel_agent", "instruction": text})
    if not stage:
        stage.append({"agent": "qa_agent", "instruction": text})
    return [stage]


def _validate_plan(result: dict, fallback_text: str) -> list[list[dict]]:
    """校验 LLM 计划：agent 名合法、阶段数/任务数截断。"""
    stages: list[list[dict]] = []
    for stage in (result.get("stages") or [])[:3]:
        if isinstance(stage, dict):
            stage = [stage]  # LLM 偶尔把单任务阶段输出成对象而非列表
        if not isinstance(stage, list):
            continue
        tasks = []
        for t in (stage or [])[:2]:
            if isinstance(t, dict) and t.get("agent") in VALID_AGENTS:
                tasks.append({
                    "agent": t["agent"],
                    "instruction": (t.get("instruction") or fallback_text).strip(),
                })
        if tasks:
            stages.append(tasks)
    return stages


def node_supervisor(state: AgentState) -> AgentState:
    """调度节点：首次进入生成计划；后续进入（阶段推进）为 no-op 放行。"""
    if state.metadata.get("guardrail_blocked"):
        return state

    # 已有计划 → 阶段推进中，由 router 决定下一跳
    if state.metadata.get("plan_stages") is not None:
        return state

    text = state.user_input or ""
    stages: list[list[dict]] = []

    if is_llm_available():
        prompt = f"用户请求：{text}"
        profile_hint = state.metadata.get("profile_hint", "")
        if profile_hint:
            prompt += f"\n（用户画像：{profile_hint.strip()}）"
        result = chat_json(prompt, system=SUPERVISOR_PROMPT, intent="")
        stages = _validate_plan(result, text)
        # 安全网：实时数据请求不允许只派 qa_agent（LLM 误判时强制走关键词规划）
        if stages and all(t["agent"] == "qa_agent" for stage in stages for t in stage):
            if _has_realtime_intent(text):
                stages = []
    if not stages:
        stages = _plan_by_keywords(text)

    state.metadata["plan_stages"] = stages
    state.metadata["stage_idx"] = 0
    state.metadata["stage_dispatched"] = False
    state.metadata["stage_done"] = []
    state.metadata["handoffs"] = []
    plan_desc = " → ".join(
        "+".join(t["agent"] for t in stage) + ("(并行)" if len(stage) > 1 else "")
        for stage in stages
    )
    state.thinking.append(f"[调度Agent] 生成执行计划：{plan_desc}")
    state.metadata["supervisor_plan"] = json.dumps(stages, ensure_ascii=False)
    return state


def router_supervisor(state: AgentState) -> str | list[str]:
    """顶层条件路由：派发当前阶段（多任务返回节点列表触发并行），或推进阶段。"""
    if state.metadata.get("guardrail_blocked"):
        return "guardrail_output"

    stages = state.metadata.get("plan_stages") or []
    idx = state.metadata.get("stage_idx", 0)

    # 全部阶段完成（可能还有运行中追加的 handoff 阶段）
    if idx >= len(stages):
        handoffs = state.metadata.get("handoffs") or []
        if handoffs:
            # 动态移交：把子 Agent 移交的请求追加为新阶段
            state.metadata["plan_stages"] = stages + [[h] for h in handoffs]
            state.metadata["handoffs"] = []
            state.thinking.append(f"[调度Agent] 接收 Agent 移交任务：{handoffs[-1]}")
            return "supervisor"  # 重新进入路由，派发新阶段
        return "guardrail_output"

    # 当前阶段已派发过（fan-out 合并后重新路由）→ 推进到下一阶段
    if state.metadata.get("stage_dispatched"):
        state.metadata["stage_dispatched"] = False
        state.metadata["stage_done"] = []
        state.metadata["stage_idx"] = idx + 1
        if state.metadata["stage_idx"] < len(state.metadata["plan_stages"]):
            return "supervisor"  # 回到本节点，派发下一阶段
        return router_supervisor(state)  # 递归判断 handoffs / 完成

    # 首次派发当前阶段
    stage = stages[idx]
    agents = []
    stage_tasks: dict[str, dict] = {}
    for t in stage:
        if t["agent"] in stage_tasks:
            stage_tasks[t["agent"]]["instruction"] += "\n" + t["instruction"]
        else:
            stage_tasks[t["agent"]] = dict(t)
            agents.append(t["agent"])
    state.metadata["stage_tasks"] = stage_tasks
    state.metadata["stage_dispatched"] = True
    if len(agents) == 1:
        return agents[0]
    return agents  # 列表 → 引擎并行 fan-out


def route_after_specialist(state: AgentState) -> str:
    """子 Agent（串行模式）完成后的路由。"""
    if state.interrupted:
        return "__end__"  # HITL 中断，返回等待确认
    # 记录本阶段完成进度
    done = state.metadata.setdefault("stage_done", [])
    done.append(state.metadata.get("current_agent", ""))
    return "supervisor"  # 回调度器推进阶段（node_supervisor 会 no-op 放行）


def pop_task(state: AgentState) -> dict:
    """串行模式兼容：从旧式任务队列弹出（并行模式走 stage_tasks）。"""
    if state.tasks:
        return state.tasks.pop(0)
    return {}
