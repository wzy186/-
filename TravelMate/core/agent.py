"""TravelMate 多 Agent 编排入口（对外 API，保持与旧版单 Agent 完全兼容）。

架构：
    Supervisor 调度 Agent → 4 个专家子 Agent（ReAct 子图）→ 结果合并
    子 Agent 工具走统一 MCP 注册表（本地工具 + 外部 MCP Server 动态注册）

对外接口：
    process(user_input, session_id) -> dict   # 与旧版字段完全一致
    resume(session_id, confirmation) -> dict  # HITL 确认后续跑
"""

from __future__ import annotations

import time

from core.graph import AgentState, CompiledGraph, _diff_state
from core.mcp import execute_tool


def _build_graph() -> CompiledGraph:
    # 延迟导入：等依赖模块都加载完成后再组装图
    from core.agents import build_orchestrator
    return build_orchestrator()


_graph = _build_graph()

# 存放被 HITL 中断的状态，按会话隔离
_interrupted_states: dict[str, AgentState] = {}


# ── Public API ──

def process(user_input: str, session_id: str = "default") -> dict:
    """处理用户输入：调度 → 子 Agent 协作 → 合并回复。"""
    state = AgentState(user_input=user_input, session_id=session_id)
    result = _graph.invoke(state)
    if result.interrupted:
        _interrupted_states[session_id] = result
    return _state_to_response(result)


def resume(session_id: str = "default", confirmation: str = "yes") -> dict:
    """HITL 中断后的恢复：确认则执行挂起操作，否则取消，然后继续收尾。"""
    state = _interrupted_states.pop(session_id, None)
    if state is None:
        return {"reply": "无需确认的操作", "tool_calls": [], "thinking": [], "actions": [],
                "interrupted": False, "interrupt_data": {}, "trace": [], "error": ""}

    state.interrupted = False
    if confirmation.lower() in ("yes", "y", "确认", "是", "ok"):
        state.needs_confirmation = False
        state = _execute_pending_actions(state)
    else:
        state.reply = "操作已取消。"
        state.pending_actions = []
        state.needs_confirmation = False

    # 从输出护栏继续收尾（子 Agent 的回答已写入 state.reply / replies）
    current = state.metadata.get("resume_node", "guardrail_output")
    for _step in range(10):
        if current == "__end__" or current not in _graph.graph.nodes:
            break
        node_func = _graph.graph.nodes[current]
        t0 = time.time()
        prev_state = state.to_dict()
        state = node_func(state)
        elapsed = time.time() - t0
        state.trace.append({"node": current, "elapsed_ms": round(elapsed * 1000, 1),
                            "state_changes": _diff_state(prev_state, state.to_dict())})
        if state.interrupted:
            break
        if current in _graph.graph.conditional_edges:
            router, mapping = _graph.graph.conditional_edges[current]
            current = mapping.get(router(state), "__end__")
        elif current in _graph.graph.edges:
            current = _graph.graph.edges[current]
        else:
            current = "__end__"
    return _state_to_response(state)


def _execute_pending_actions(state: AgentState) -> AgentState:
    """执行等待 HITL 确认的操作，并把结果追加到回复。"""
    for pa in state.pending_actions:
        result = execute_tool(pa["tool"], pa["args"])
        state.actions.append({"tool": pa["tool"], "args": pa["args"],
                              "agent": pa.get("agent", ""),
                              "result": result.data or {}})
        if result.message:
            state.thinking.append(f"[执行] {result.message[:100]}")
            state.reply = (state.reply or "") + ("\n\n" if state.reply else "") + result.message
    state.pending_actions = []
    return state


def _state_to_response(state: AgentState) -> dict:
    return {
        "reply": state.reply,
        "tool_calls": state.tool_calls,
        "thinking": state.thinking,
        "actions": state.actions,
        "interrupted": state.interrupted,
        "interrupt_data": state.interrupt_data,
        "trace": state.trace,
        "error": state.error,
    }
