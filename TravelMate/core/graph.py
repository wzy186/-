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
    # Tracing
    trace: list[dict] = field(default_factory=list)

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

    def add_edge(self, from_node: str, to_node: str):
        self.edges[from_node] = to_node

    def add_conditional_edges(self, from_node: str, router: RouterFunc, mapping: dict[str, str]):
        self.conditional_edges[from_node] = (router, mapping)

    def set_entry_point(self, name: str):
        self.entry_point = name

    def compile(self) -> "CompiledGraph":
        return CompiledGraph(self)


class CompiledGraph:
    """Executable compiled graph."""

    def __init__(self, graph: StateGraph):
        self.graph = graph

    def invoke(self, initial_state: AgentState) -> AgentState:
        """Run the graph to completion (or until interrupt)."""
        state = initial_state
        current = self.graph.entry_point

        max_steps = 20
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
                current = mapping.get(next_node, "__end__")
            elif current in self.graph.edges:
                current = self.graph.edges[current]
            else:
                current = "__end__"

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


def _diff_state(prev: dict, curr: dict) -> dict:
    """Compute which fields changed between two state dicts."""
    changes = {}
    for key in curr:
        if key not in prev or prev[key] != curr[key]:
            if key == "trace":
                continue  # Skip trace itself
            changes[key] = True
    return changes
