"""顶层编排图的通用节点：上下文加载、护栏、收尾。"""

from __future__ import annotations

from core.graph import AgentState
from core.guardrails import (
    has_critical_failure, has_warnings, run_input_guardrails, run_output_guardrails,
)
from core.memory import add_session_message, get_profile, get_session
from core.prompts import SYSTEM_PROMPT
from core.rag import get_context


def node_load_context(state: AgentState) -> AgentState:
    """加载用户画像 + RAG 上下文 + 会话历史。"""
    profile = get_profile()
    profile_hint = ""
    if profile:
        parts = [f"{k}: {v}" for k, v in profile.items() if v]
        if parts:
            profile_hint = f"用户偏好画像：{', '.join(parts)}"
    context = get_context(state.user_input)
    state.metadata["profile_hint"] = profile_hint
    state.metadata["rag_context"] = context or ""
    state.metadata["system_prompt"] = SYSTEM_PROMPT + (
        f"\n{profile_hint}" if profile_hint else ""
    ) + (f"\n{context}" if context else "")

    history = get_session(state.session_id)
    add_session_message(state.session_id, "user", state.user_input)
    messages = [{"role": m["role"], "content": m["content"]} for m in history[-12:]]
    messages.append({"role": "user", "content": state.user_input})
    state.messages = messages
    return state


def node_guardrail_input(state: AgentState) -> AgentState:
    results = run_input_guardrails(state.user_input)
    failure = has_critical_failure(results)
    if failure:
        state.reply = f"⚠️ 输入安全检查未通过: {failure.reason}。请修改您的问题。"
        state.error = failure.reason
        state.metadata["guardrail_blocked"] = True
        return state
    warnings = has_warnings(results)
    if warnings:
        state.metadata["guardrail_warnings"] = warnings
    return state


def node_guardrail_output(state: AgentState) -> AgentState:
    if state.metadata.get("guardrail_blocked"):
        return state
    results = run_output_guardrails(state.reply)
    failure = has_critical_failure(results)
    if failure:
        state.reply = "⚠️ 输出安全检查未通过，已过滤。请重新提问。"
        state.error = failure.reason
    warnings = state.metadata.get("guardrail_warnings", [])
    if warnings:
        state.reply += "\n\n" + "\n".join(f"⚠️ {w}" for w in warnings)
    return state


def node_finalize(state: AgentState) -> AgentState:
    """收尾：多任务时合并各子 Agent 的回答，保存会话。"""
    replies = state.metadata.get("replies", [])
    if len(replies) > 1:
        # 多任务：每个子 Agent 的回答带标题合并
        parts = []
        for r in replies:
            parts.append(f"### {r['agent']}\n{r['reply']}")
        state.reply = "\n\n---\n\n".join(parts)
    if not state.reply:
        state.reply = "已完成您的请求！"
    # 清理本轮迭代计数，避免影响下一轮对话
    for key in [k for k in state.metadata if k.startswith("__iter_")]:
        state.metadata.pop(key, None)
    state.metadata.pop("current_task", None)
    add_session_message(state.session_id, "assistant", state.reply)
    return state
