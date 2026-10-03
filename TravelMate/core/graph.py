"""Lightweight StateGraph engine — inspired by LangGraph.
Supports: nodes, edges, conditional routing, state persistence, interrupt/resume."""

from __future__ import annotations
import json
from typing import Any, Callable
from dataclasses import dataclass, field


@dataclass
class AgentState:
    """State that flows through the graph."""
    messages: list[dict] = field(default_factory=list)
    user_input: str = ""
    session_id: str = "default"
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    reply: str = ""
    interrupted: bool = False
    interrupt_data: dict = field(default_factory=dict)
    needs_confirmation: bool = False
    pending_actions: list[dict] = field(default_factory=list)
    error: str = ""
    metadata: dict = field(default_factory=dict)
    # 多 Agent 调度：supervisor 派发的子任务队列 [{"agent": str, "instruction": str}]
    tasks: list[dict] = field(default_factory=list)
    # Tracing
    trace: list[dict] = field(default_factory=list)

    def clone(self) -> "AgentState":
        """轻量克隆（用于并行分支）：容器层浅拷贝。

        分支只做追加/弹出容器操作，不修改内部共享对象，因此无需 deepcopy
        （deepcopy 在此状态上开销大且有线程问题）。
        """
        return AgentState(
            messages=list(self.messages),
            user_input=self.user_input,
            session_id=self.session_id,
            tool_calls=list(self.tool_calls),
            tool_results=list(self.tool_results),
            thinking=list(self.thinking),
            actions=list(self.actions),
            reply=self.reply,
            interrupted=False,
            interrupt_data={},
            needs_confirmation=self.needs_confirmation,
            pending_actions=list(self.pending_actions),
            error=self.error,
            metadata={
                k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
                for k, v in self.metadata.items()
            },
            tasks=list(self.tasks),
            trace=[],
        )

    def to_dict(self) -> dict:
        return {
            "messages": self.messages, "user_input": self.user_input,
            "session_id": self.session_id, "tool_calls": self.tool_calls,
            "tool_results": self.tool_results, "thinking": self.thinking,
            "actions": self.actions, "reply": self.reply,
            "interrupted": self.interrupted, "interrupt_data": self.interrupt_data,
            "needs_confirmation": self.needs_confirmation,
            "pending_actions": self.pending_actions, "error": self.error,
            "metadata": self.metadata, "trace": self.trace,
            "tasks": self.tasks,
        }


NodeFunc = Callable[[AgentState], AgentState]
RouterFunc = Callable[[AgentState], str]


class StateGraph:
    """Directed graph with conditional edges for agent orchestration."""

    def __init__(self, state_class=AgentState):
        self.nodes: dict[str, NodeFunc] = {}
        self.edges: dict[str, str] = {}
        self.conditional_edges: dict[str, tuple[RouterFunc, dict[str, str]]] = {}
        self.entry_point: str = ""
        self._state_class = state_class

    def add_node(self, name: str, func: NodeFunc):
        self.nodes[name] = func

    def add_subgraph(self, name: str, subgraph: "CompiledGraph"):
        """将一个编译好的子图（子 Agent）作为节点嵌入当前图。

        父图与子图共享同一个 AgentState；子图执行产生的 thinking / trace /
        tool_results / interrupt 都直接反映在共享状态上。
        子图内部发生 HITL 中断时，状态带 interrupted=True 返回，
        由父图的条件边或引擎中断检查接管。
        """

        def _run_subgraph(state: AgentState) -> AgentState:
            return subgraph.invoke(state)

        self.nodes[name] = _run_subgraph

    def add_edge(self, from_node: str, to_node: str):
        self.edges[from_node] = to_node

    def add_conditional_edges(self, from_node: str, router: RouterFunc, mapping: dict[str, str]):
        self.conditional_edges[from_node] = (router, mapping)

    def set_entry_point(self, name: str):
        self.entry_point = name

    def compile(self) -> "CompiledGraph":
        return CompiledGraph(self)


class CompiledGraph:
    """Executable compiled graph.

    动态多 Agent 支持：
    - 条件路由函数可返回单个 key（串行）或 key 列表（并行 fan-out）
    - 并行分支各自在 state 的深拷贝上独立执行，结束后按字段合并回主状态
      （列表字段拼接、metadata 深合并、interrupt/error/reply 聚合）
    - fan-out 完成后会再次评估同一节点的条件路由，由路由函数决定下一跳
      （路由函数应通过 metadata 标记避免重复派发）
    """

    def __init__(self, graph: StateGraph):
        self.graph = graph

    def invoke(self, initial_state: AgentState) -> AgentState:
        """Run the graph to completion (or until interrupt)."""
        state = initial_state
        current = self.graph.entry_point

        max_steps = 40
        for step in range(max_steps):
            if current == "__end__" or current not in self.graph.nodes:
                break

            # Execute node
            node_func = self.graph.nodes[current]
            import time
            t0 = time.time()
            prev_state = state.to_dict()
            state = node_func(state)
            elapsed = time.time() - t0

            # Record trace
            state.trace.append({
                "node": current,
                "elapsed_ms": round(elapsed * 1000, 1),
                "state_changes": _diff_state(prev_state, state.to_dict()),
            })

            # Check for interrupt
            if state.interrupted:
                return state

            # Determine next node
            if current in self.graph.conditional_edges:
                router, mapping = self.graph.conditional_edges[current]
                next_node = router(state)
                if isinstance(next_node, (list, tuple)) and len(next_node) > 1:
                    # 并行 fan-out：多分支同时执行，完成后合并并重新路由
                    state = self._fanout(state, list(next_node))
                    state.trace.append({
                        "node": f"__parallel__({', '.join(next_node)})",
                        "elapsed_ms": 0.0,
                        "state_changes": {},
                    })
                    if state.interrupted:
                        return state
                    # 重新评估同一节点的条件路由（router 内部推进阶段标记）
                    next_node = router(state)
                    if isinstance(next_node, (list, tuple)):
                        next_node = next_node[0]  # 防御：不允许连续两次 fan-out
                next_node = next_node[0] if isinstance(next_node, (list, tuple)) else next_node
                current = mapping.get(next_node, "__end__")
            elif current in self.graph.edges:
                current = self.graph.edges[current]
            else:
                current = "__end__"

        return state

    def _fanout(self, state: AgentState, names: list[str]) -> AgentState:
        """多分支分发：各分支在状态克隆上依次执行，结果合并回主状态。

        注：为保证稳定性采用顺序分发（多线程版在 deepcopy/json 场景有卡死问题）。
        多 Agent 的动态性由 Plan-and-Execute 阶段调度 + handoff 移交提供。
        """
        trace_offset = len(state.trace)
        fork = {
            "tool_calls": len(state.tool_calls),
            "tool_results": len(state.tool_results),
            "thinking": len(state.thinking),
            "actions": len(state.actions),
            "metadata_lists": {k: len(v) for k, v in state.metadata.items() if isinstance(v, list)},
        }
        for name in names:
            branch_state = state.clone()
            try:
                branch_result = self.graph.nodes[name](branch_state)
                _merge_state(state, branch_result, trace_from=trace_offset, fork=fork)
            except Exception as e:  # 单分支失败不影响其他分支
                state.thinking.append(f"[分支 {name}] ❌ 执行失败: {str(e)[:120]}")
            trace_offset = len(state.trace)

        state.thinking.append(f"[分发] 完成：{' + '.join(names)}")
        return state

    def resume(self, state: AgentState, user_response: str = "yes") -> AgentState:
        """Resume from an interrupt with user's confirmation."""
        if not state.interrupted:
            return state

        state.interrupted = False
        if user_response.lower() in ("yes", "y", "确认", "是", "ok"):
            state.needs_confirmation = False
            # Execute the pending actions
            from core.agent import _execute_pending_actions
            state = _execute_pending_actions(state)
        else:
            state.reply = "操作已取消。"
            state.pending_actions = []
            state.needs_confirmation = False

        # Continue the graph from where we left off
        if "resume_node" in state.metadata:
            current = state.metadata["resume_node"]
        else:
            return state

        max_steps = 10
        for step in range(max_steps):
            if current == "__end__" or current not in self.graph.nodes:
                break

            node_func = self.graph.nodes[current]
            import time
            t0 = time.time()
            prev_state = state.to_dict()
            state = node_func(state)
            elapsed = time.time() - t0
            state.trace.append({
                "node": current,
                "elapsed_ms": round(elapsed * 1000, 1),
                "state_changes": _diff_state(prev_state, state.to_dict()),
            })

            if state.interrupted:
                return state

            if current in self.graph.conditional_edges:
                router, mapping = self.graph.conditional_edges[current]
                next_node = router(state)
                current = mapping.get(next_node, "__end__")
            elif current in self.graph.edges:
                current = self.graph.edges[current]
            else:
                current = "__end__"

        return state


def _merge_state(base: AgentState, branch: AgentState, trace_from: int = 0, fork: dict | None = None):
    """把分支的状态合并回主状态——只合并分支产生的增量。

    分支是主状态的克隆，携带 fork 前的全部数据；直接拼接会导致列表翻倍
    （plan_stages 翻倍曾引发无限重复派发），因此按 fork 点长度截取增量。
    """
    fork = fork or {}

    def _delta(field: str) -> list:
        start = fork.get(field, 0)
        return getattr(branch, field)[start:]

    base.tool_calls.extend(_delta("tool_calls"))
    base.tool_results.extend(_delta("tool_results"))
    base.thinking.extend(_delta("thinking"))
    base.actions.extend(_delta("actions"))
    base.trace.extend(branch.trace[trace_from:])

    fork_meta = fork.get("metadata_lists", {})
    bm = base.metadata
    for k, v in branch.metadata.items():
        if isinstance(v, list) and k in fork_meta:
            delta = v[fork_meta[k]:]
            if delta:
                bm.setdefault(k, [])
                if isinstance(bm[k], list):
                    bm[k].extend(delta)
                else:
                    bm[k] = delta
        elif k not in bm:
            bm[k] = v
        # 分支继承且未新增的键：以 base 为准，跳过

    if branch.interrupted:
        base.interrupted = True
        base.needs_confirmation = True
        if branch.interrupt_data:
            base.interrupt_data = branch.interrupt_data
        base.pending_actions = (base.pending_actions or []) + (branch.pending_actions or [])
    if branch.error and not base.error:
        base.error = branch.error
    if branch.reply and not base.reply:
        base.reply = branch.reply


def _diff_state(prev: dict, curr: dict) -> dict:
    """Compute which fields changed between two state dicts."""
    changes = {}
    for key in curr:
        if key not in prev or prev[key] != curr[key]:
            if key == "trace":
                continue  # Skip trace itself
            changes[key] = True
    return changes
