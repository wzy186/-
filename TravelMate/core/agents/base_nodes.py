"""子 Agent 基础设施 — 每个专家 Agent 是一个独立的 ReAct 子图。

SpecialistConfig 定义一个子 Agent 的：系统提示词、工具白名单、远端 MCP 前缀、
最大推理轮数、Mock 行为。build_specialist_graph 将其编译为 CompiledGraph，
由顶层编排器通过 StateGraph.add_subgraph 嵌入。

子图结构（ReAct 循环，最多 max_iter 轮）：
    react_llm → parse_tools ─(无工具调用)→ agent_format → 结束
                     └─(有工具调用)→ react_exec ─(需继续推理)→ react_llm
                                              └─(HITL 中断)→ 中断返回父图
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from core.graph import AgentState, CompiledGraph, StateGraph
from core.llm import chat, is_llm_available
from core.mcp import execute_tool, get_all_tool_schemas, get_tool_schema

# 兼容多种 LLM 的工具调用输出格式：
# 1. 约定格式:      [call:toolName] {json}
# 2. DeepSeek DSML: <｜｜DSML｜｜ invoke name="tool"> {json}
# 3. 通用 invoke:   <invoke name="tool"> {json}
# 4. function 标记: <function=tool> {json}
_CALL_PATTERNS = [
    re.compile(r"\[call:(\w+)\]\s*(\{[^}]*\})?", re.S),
    re.compile(r"<｜｜DSML｜｜\s*invoke\s+name=[\"'](\w+)[\"']\s*>\s*(\{[\s\S]*?\})?", re.I),
    re.compile(r"<invoke\s+name=[\"'](\w+)[\"']\s*>\s*(\{[\s\S]*?\})?", re.I),
    re.compile(r"<function\s*=\s*[\"']?(\w+)[\"']?\s*>\s*(\{[\s\S]*?\})?", re.I),
]
# 回复中需要清除的杂项标记（含闭合标签）
_LEFTOVER_TAGS = re.compile(
    r"</?｜｜DSML｜｜[^>]*>|</invoke>|</function[^>]*>|<｜｜DSML｜｜>", re.S)


def parse_tool_calls(text: str) -> tuple[list[dict], str]:
    """从 LLM 回复中解析工具调用，返回 (calls, 清理后的纯文本回复)。"""
    calls: list[dict] = []
    spans: list[tuple[int, int]] = []
    for pat in _CALL_PATTERNS:
        for m in pat.finditer(text):
            if any(s <= m.start() < e for s, e in spans):
                continue  # 与已匹配区间重叠（如 DSML 被 invoke 模式重复命中）
            args = {}
            if m.group(2):
                try:
                    args = json.loads(m.group(2))
                except json.JSONDecodeError:
                    args = {}
            calls.append({"tool": m.group(1), "args": args})
            spans.append(m.span())
    # 按出现顺序排列，并从回复中剥离
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    calls = [calls[i] for i in order]
    parts, last = [], 0
    for s, e in sorted(spans):
        parts.append(text[last:s])
        last = e
    parts.append(text[last:])
    cleaned = _LEFTOVER_TAGS.sub("", "".join(parts)).strip()
    return calls, cleaned


@dataclass
class SpecialistConfig:
    """一个专家子 Agent 的声明式配置。"""
    name: str                       # 子图节点名（英文）
    label: str                      # 展示名（中文）
    system_prompt: str              # 专属系统提示词
    tools: set = field(default_factory=set)          # 本地工具白名单
    remote_prefixes: tuple = ()     # 远端 MCP 工具名前缀，如 ("mcp_amap",)
    max_iter: int = 3               # ReAct 最大轮数
    mock: callable = None           # Mock 模式处理器 (state, instruction, iters) -> str
    mock_intent: str = ""           # 无 LLM 时 chat() 的 intent 提示
    ensure_action: callable = None  # 确定性动作兜底 (state) -> bool：意图明确但LLM未发起操作时自动构造


def tool_allowed(cfg: SpecialistConfig, tool_name: str) -> bool:
    if tool_name in cfg.tools:
        return True
    return any(tool_name.startswith(p) for p in cfg.remote_prefixes)


def tool_lines_for(cfg: SpecialistConfig) -> str:
    """生成属于本 Agent 的工具说明（含远端 MCP 动态注册的工具）。"""
    lines = []
    for schema in get_all_tool_schemas():
        if not tool_allowed(cfg, schema.name):
            continue
        cat = "操作" if schema.category == "action" else "查询"
        params = ", ".join(
            f'"{k}": {v.get("description", "")}'
            for k, v in (schema.parameters.get("properties") or {}).items()
        )
        lines.append(f"- {schema.name}: {schema.description} [{cat}] {{{params}}}")
    return "\n".join(lines)


def build_specialist_graph(cfg: SpecialistConfig) -> CompiledGraph:
    """把专家配置编译成可执行的 ReAct 子图。"""

    # ── 子图节点 ──

    def react_llm(state: AgentState) -> AgentState:
        if state.metadata.get("guardrail_blocked"):
            return state
        iters = state.metadata.get(f"__iter_{cfg.name}", 0)
        state.metadata[f"__iter_{cfg.name}"] = iters + 1

        # 首轮进入：领取调度任务（并行模式按 agent 名领取，串行模式从队列弹出）
        if iters == 0:
            from core.agents.supervisor import pop_task
            stage_tasks = state.metadata.get("stage_tasks") or {}
            my_task = stage_tasks.pop(cfg.name, None)
            task = my_task or pop_task(state)
            state.metadata["current_task"] = task
            state.metadata["current_agent"] = cfg.label
            state.thinking.append(f"[进入子Agent] {cfg.label}")

        system = cfg.system_prompt
        tool_lines = tool_lines_for(cfg)
        if tool_lines:
            system += (
                "\n\n## 你的专属工具（只能使用这些工具）\n" + tool_lines +
                '\n\n## 工具调用格式（严格遵守）\n[call:toolName] {"param1": "value1"}\n'
                "规则：\n1. 需要数据时先调用查询工具；2. 拿到结果后综合成完整回答，"
                "不要把工具调用语法留在最终回复里；3. 用户表达操作意图（预订等）时直接调用操作工具。\n"
                "4. 工具调用只允许使用 [call:toolName] {json} 这一种格式，"
                "严禁使用 XML 标签、函数调用标记或其他任何格式。\n"
                "5. 若执行中发现需要其他专家处理的子请求（如路线Agent遇到订票需求），"
                "调用 [call:handoff] {\"agent\": \"目标专家名\", \"instruction\": \"子请求\"} 移交，"
                "然后继续完成自己的部分。可选目标: route_agent/ticket_agent/travel_agent/qa_agent。"
            )

        task = state.metadata.get("current_task") or {}
        instruction = task.get("instruction") or state.user_input

        messages = list(state.messages)
        messages.append({"role": "user", "content": instruction})

        # 第 2+ 轮：把上一轮工具结果喂回给 LLM 综合
        if iters > 0 and state.tool_results:
            recent = state.tool_results[-4:]
            tr_text = "\n".join(
                f"[{t['tool']}] {json.dumps(t.get('result', {}), ensure_ascii=False)[:600]}"
                for t in recent
            )
            messages.append({
                "role": "user",
                "content": f"工具查询结果：\n{tr_text}\n\n请基于以上结果继续：若数据已足够，"
                           f"直接给出最终回答（不要再输出工具调用语法）。",
            })

        if is_llm_available():
            prompt = "\n".join(
                f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}" for m in messages
            )
            reply = chat(prompt, system, intent=cfg.mock_intent)
        else:
            reply = cfg.mock(state, instruction, iters) if cfg.mock else (
                "我是{label}，当前为 Mock 模式。".format(label=cfg.label)
            )

        state.reply = reply
        state.metadata["raw_reply"] = reply
        return state

    def parse_tools(state: AgentState) -> AgentState:
        """解析多格式工具调用语法（约定格式/DeepSeek DSML/通用 invoke）并剥离。"""
        calls, clean = parse_tool_calls(state.reply or "")
        state.tool_calls = calls
        state.reply = clean
        return state

    def react_exec(state: AgentState) -> AgentState:
        if state.metadata.get("guardrail_blocked"):
            return state
        pending = []
        for tc in state.tool_calls:
            name, args = tc["tool"], tc["args"]
            # Agent 间动态移交：发现超出职责的子请求，交回调度器追加执行阶段
            if name == "handoff":
                from core.agents.supervisor import VALID_AGENTS
                target = (args or {}).get("agent", "")
                if target in VALID_AGENTS:
                    handoff_task = {"agent": target,
                                    "instruction": (args or {}).get("instruction") or state.user_input}
                    state.metadata.setdefault("handoffs", []).append(handoff_task)
                    state.thinking.append(f"[{cfg.label}] ⤴️ 移交给 {target}：{handoff_task['instruction'][:60]}")
                else:
                    state.thinking.append(f"[{cfg.label}] ⚠️ 无效移交目标: {target}")
                continue
            if not tool_allowed(cfg, name):
                state.thinking.append(f"[{cfg.label}] ⚠️ 工具 {name} 不属于本 Agent，已跳过")
                continue
            state.thinking.append(f"[{cfg.label}] 调用工具 {name}({json.dumps(args, ensure_ascii=False)[:80]})")
            schema = get_tool_schema(name)
            if schema and schema.category == "action":
                # HITL：操作类工具先挂起，等用户确认
                pending.append({"tool": name, "args": args, "agent": cfg.name})
            else:
                result = execute_tool(name, args)
                state.tool_results.append({
                    "tool": name, "args": args, "agent": cfg.name,
                    "result": result.data or {}, "is_action": False,
                })
                msg = result.message or (f"错误: {result.error}" if result.error else "完成")
                state.thinking.append(f"[{cfg.label}] {msg[:100]}")

        if pending:
            state.pending_actions = pending
            state.needs_confirmation = True
            desc = "\n".join(
                f"• {a['tool']}: {json.dumps(a['args'], ensure_ascii=False)[:100]}"
                for a in pending
            )
            state.interrupted = True
            state.interrupt_data = {
                "type": "action_confirmation",
                "actions": pending,
                "message": f"⚠️ {cfg.label}请求执行以下操作，请确认：\n{desc}",
            }
            state.metadata["resume_node"] = "guardrail_output"
        return state

    def agent_format(state: AgentState) -> AgentState:
        """整理本 Agent 的最终回答；回复过短时用工具结果拼装。"""
        # 确定性动作兜底：用户意图明确但 LLM 只说不做 → 直接构造操作进 HITL 确认框
        if cfg.ensure_action and cfg.ensure_action(state):
            return state
        reply = (state.reply or "").strip()
        if len(reply) < 10 and state.tool_results:
            parts = []
            for tr in state.tool_results:
                data = tr.get("result", {})
                if isinstance(data, dict) and data.get("message"):
                    parts.append(data["message"])
            reply = "\n\n".join(parts) if parts else f"{cfg.label}已完成查询。"
        state.reply = reply
        state.metadata.setdefault("replies", []).append({"agent": cfg.label, "reply": reply})
        return state

    # ── 子图路由 ──

    def route_after_parse(state: AgentState) -> str:
        if state.metadata.get("guardrail_blocked"):
            return "agent_format"
        return "react_exec" if state.tool_calls else "agent_format"

    def route_after_exec(state: AgentState) -> str:
        if state.interrupted:
            return "__end__"  # HITL 中断，返回父图
        iters = state.metadata.get(f"__iter_{cfg.name}", 0)
        # 还有推理预算且有新结果 → 回 LLM 综合回答
        if iters < cfg.max_iter and state.tool_calls:
            return "react_llm"
        return "agent_format"

    # ── 组装子图 ──

    g = StateGraph(AgentState)
    g.add_node("react_llm", react_llm)
    g.add_node("parse_tools", parse_tools)
    g.add_node("react_exec", react_exec)
    g.add_node("agent_format", agent_format)
    g.set_entry_point("react_llm")
    g.add_edge("react_llm", "parse_tools")
    g.add_conditional_edges("parse_tools", route_after_parse, {
        "react_exec": "react_exec", "agent_format": "agent_format",
    })
    g.add_conditional_edges("react_exec", route_after_exec, {
        "react_llm": "react_llm", "agent_format": "agent_format",
    })
    return g.compile()
