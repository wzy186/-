"""TravelMate ReAct Agent — rebuilt with StateGraph, Guardrails, MCP, HITL, Tracing."""

import json
import re
from core.graph import StateGraph, AgentState, CompiledGraph, _diff_state
from core.llm import chat, chat_json, is_llm_available, _mock
from core.prompts import SYSTEM_PROMPT
from core.rag import get_context
from core.memory import get_profile, add_session_message, get_session
from core.guardrails import run_input_guardrails, run_output_guardrails, has_critical_failure, has_warnings
from core.mcp import execute_tool, get_tool_schema, get_tools_for_prompt

MAX_ITERATIONS = 6

# ── Graph Nodes ──

def node_load_context(state: AgentState) -> AgentState:
    """Load user profile and RAG context."""
    profile = get_profile()
    profile_hint = ""
    if profile:
        parts = [f"{k}: {v}" for k, v in profile.items() if v]
        if parts:
            profile_hint = f"\n用户偏好画像：{', '.join(parts)}"

    context = get_context(state.user_input)
    state.metadata["system_prompt"] = SYSTEM_PROMPT + profile_hint + ("\n" + context if context else "")

    history = get_session(state.session_id)
    add_session_message(state.session_id, "user", state.user_input)

    messages = []
    for m in history[-12:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": state.user_input})
    state.messages = messages
    return state


def node_guardrail_input(state: AgentState) -> AgentState:
    """Run input guardrails."""
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


def node_llm_call(state: AgentState) -> AgentState:
    """Call LLM with conversation history."""
    if state.metadata.get("guardrail_blocked"):
        return state

    system = state.metadata.get("system_prompt", SYSTEM_PROMPT)

    if not is_llm_available():
        reply = _mock("", state.user_input)
    else:
        reply = _call_with_history(system, state.messages)

    state.reply = reply
    state.metadata["raw_reply"] = reply
    return state


def node_parse_tools(state: AgentState) -> AgentState:
    """Parse tool calls from LLM reply."""
    if state.metadata.get("guardrail_blocked"):
        return state

    reply = state.reply
    tcs = _parse_all_tool_calls(reply)
    state.tool_calls = tcs
    state.reply = _clean_reply(reply)
    return state


def node_execute_tools(state: AgentState) -> AgentState:
    """Execute parsed tool calls. Action tools require confirmation (HITL)."""
    if state.metadata.get("guardrail_blocked"):
        return state

    if not state.tool_calls:
        return state

    pending_actions = []
    for tc in state.tool_calls:
        tool_name = tc["tool"]
        args = tc["args"]
        state.thinking.append(f"[步骤] 调用工具 {tool_name}({json.dumps(args, ensure_ascii=False)[:80]})")

        # Check if this is an action tool
        schema = get_tool_schema(tool_name)
        is_action = schema and schema.category == "action"

        if is_action:
            # HITL: queue for confirmation instead of executing immediately
            pending_actions.append({"tool": tool_name, "args": args})
        else:
            # Query tool: execute immediately
            result = execute_tool(tool_name, args)
            state.tool_results.append({"tool": tool_name, "args": args, "result": result.data or {}, "is_action": False})
            if result.message:
                state.thinking.append(f"[结果] {result.message[:100]}")

    if pending_actions:
        state.pending_actions = pending_actions
        state.needs_confirmation = True

        # Build confirmation prompt
        action_desc = "\n".join(
            f"• {a['tool']}: {json.dumps(a['args'], ensure_ascii=False)[:100]}"
            for a in pending_actions
        )
        state.interrupted = True
        state.interrupt_data = {
            "type": "action_confirmation",
            "actions": pending_actions,
            "message": f"⚠️ 以下操作将直接执行，请确认：\n{action_desc}",
        }
        state.metadata["resume_node"] = "format_reply"

    return state


def node_format_reply(state: AgentState) -> AgentState:
    """Format the final reply with tool results and action confirmations."""
    if state.metadata.get("guardrail_blocked"):
        return state

    # If reply was only tool calls with no text
    if not state.reply.strip() or len(state.reply.strip()) < 10:
        # Generate summary from results
        parts = []
        for tr in state.tool_results:
            data = tr.get("result", {})
            if isinstance(data, dict) and data.get("message"):
                parts.append(data["message"])
        for a in state.actions:
            data = a.get("result", {})
            if isinstance(data, dict) and data.get("message"):
                parts.append(data["message"])

        if parts:
            state.reply = "\n\n".join(parts)
        elif state.tool_calls:
            state.reply = " | ".join(f"已执行 {tc['tool']}" for tc in state.tool_calls)
        else:
            state.reply = "已完成您的请求！"

    return state


def node_guardrail_output(state: AgentState) -> AgentState:
    """Run output guardrails."""
    if state.metadata.get("guardrail_blocked"):
        return state

    results = run_output_guardrails(state.reply)
    failure = has_critical_failure(results)
    if failure:
        state.reply = "⚠️ 输出安全检查未通过，已过滤。请重新提问。"
        state.error = failure.reason

    # Append guardrail warnings
    warnings = state.metadata.get("guardrail_warnings", [])
    if warnings:
        state.reply += "\n\n" + "\n".join(f"⚠️ {w}" for w in warnings)

    return state


def node_finalize(state: AgentState) -> AgentState:
    """Finalize: save to session, clean up."""
    if not state.reply:
        state.reply = "已完成您的请求！"

    add_session_message(state.session_id, "assistant", state.reply)
    return state


# ── Routing functions ──

def route_after_guardrail(state: AgentState) -> str:
    if state.metadata.get("guardrail_blocked"):
        return "finalize"
    return "llm_call"


def route_after_parse(state: AgentState) -> str:
    if not state.tool_calls:
        return "format_reply"
    return "execute_tools"


def route_after_execute(state: AgentState) -> str:
    if state.interrupted:
        return "__end__"  # Stop for HITL confirmation
    return "format_reply"


# ── Build the graph ──

def _build_graph() -> CompiledGraph:
    g = StateGraph(AgentState)
    g.add_node("load_context", node_load_context)
    g.add_node("guardrail_input", node_guardrail_input)
    g.add_node("llm_call", node_llm_call)
    g.add_node("parse_tools", node_parse_tools)
    g.add_node("execute_tools", node_execute_tools)
    g.add_node("format_reply", node_format_reply)
    g.add_node("guardrail_output", node_guardrail_output)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("load_context")

    g.add_edge("load_context", "guardrail_input")
    g.add_conditional_edges("guardrail_input", route_after_guardrail, {
        "llm_call": "llm_call", "finalize": "finalize"
    })
    g.add_edge("llm_call", "parse_tools")
    g.add_conditional_edges("parse_tools", route_after_parse, {
        "execute_tools": "execute_tools", "format_reply": "format_reply"
    })
    g.add_conditional_edges("execute_tools", route_after_execute, {
        "format_reply": "format_reply", "__end__": "__end__"
    })
    g.add_edge("format_reply", "guardrail_output")
    g.add_edge("guardrail_output", "finalize")

    return g.compile()


_graph = _build_graph()

# Store interrupted states for HITL resume
_interrupted_states: dict[str, AgentState] = {}


# ── Public API ──

def process(user_input: str, session_id: str = "default") -> dict:
    """Process user input through the agent graph."""
    state = AgentState(user_input=user_input, session_id=session_id)
    result = _graph.invoke(state)

    if result.interrupted:
        _interrupted_states[session_id] = result

    return _state_to_response(result)


def resume(session_id: str = "default", confirmation: str = "yes") -> dict:
    """Resume after an interrupt (HITL confirmation)."""
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

    # Continue graph from format_reply node
    current = "format_reply"
    for step in range(10):
        if current == "__end__" or current not in _graph.graph.nodes:
            break

        node_func = _graph.graph.nodes[current]
        import time
        t0 = time.time()
        prev_state = state.to_dict()
        state = node_func(state)
        elapsed = time.time() - t0
        state.trace.append({"node": current, "elapsed_ms": round(elapsed * 1000, 1),
                           "state_changes": _diff_state(prev_state, state.to_dict())})

        if current in _graph.graph.edges:
            current = _graph.graph.edges[current]
        else:
            current = "__end__"

    return _state_to_response(state)


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


def _execute_pending_actions(state: AgentState) -> AgentState:
    """Execute actions that were pending HITL confirmation."""
    for pa in state.pending_actions:
        result = execute_tool(pa["tool"], pa["args"])
        state.actions.append({"tool": pa["tool"], "args": pa["args"], "result": result.data or {}})
        if result.message:
            state.thinking.append(f"[执行] {result.message[:100]}")

    state.pending_actions = []
    return state


# ── Helpers ──

def _call_with_history(system: str, messages: list) -> str:
    parts = []
    for m in messages:
        if m["role"] == "user":
            parts.append(f"用户: {m['content']}")
        else:
            parts.append(f"助手: {m['content']}")
    return chat("\n".join(parts), system)


def _parse_all_tool_calls(text: str) -> list[dict]:
    results = []
    for m in re.finditer(r'\[call:(\w+)\]\s*(\{[^}]*\})?', text):
        tool = m.group(1)
        args = {}
        if m.group(2):
            try:
                args = json.loads(m.group(2))
            except json.JSONDecodeError:
                pass
        results.append({"tool": tool, "args": args})
    return results


def _clean_reply(text: str) -> str:
    return re.sub(r'\[call:\w+\]\s*(\{[^}]*\})?', '', text).strip()
