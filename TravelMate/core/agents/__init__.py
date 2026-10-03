"""多 Agent 编排层。

- supervisor: 调度 Agent（意图识别 + 任务派发）
- SPECIALISTS: 四个专家子 Agent 配置（路线/票务/行程/问答）
- build_specialist_graph: 将专家配置编译为 ReAct 子图
- build_orchestrator(): 组装顶层多 Agent 图

顶层结构：
    load_context → guardrail_input ─(拦截)→ finalize
                        │(放行)
                    supervisor ──路由──► 专家子图（route/ticket/travel/qa）
                        ▲                    │
                        │(还有任务)           │(完成/中断)
                        └─────────────── guardrail_output → finalize
"""

from __future__ import annotations

from core.agents.base_nodes import SpecialistConfig, build_specialist_graph
from core.agents.configs import SPECIALISTS
from core.graph import AgentState, CompiledGraph, StateGraph


def build_orchestrator() -> CompiledGraph:
    from core.agents.nodes import (
        node_load_context, node_guardrail_input, node_guardrail_output,
        node_finalize,
    )
    from core.agents.supervisor import (
        node_supervisor, router_supervisor, route_after_specialist,
    )

    g = StateGraph(AgentState)
    g.add_node("load_context", node_load_context)
    g.add_node("guardrail_input", node_guardrail_input)
    g.add_node("supervisor", node_supervisor)
    g.add_node("guardrail_output", node_guardrail_output)
    g.add_node("finalize", node_finalize)

    # 四个专家子 Agent 以子图形式嵌入（共享 AgentState）
    for name, cfg in SPECIALISTS.items():
        g.add_subgraph(name, build_specialist_graph(cfg))
        g.add_conditional_edges(name, route_after_specialist, {
            "supervisor": "supervisor",
            "guardrail_output": "guardrail_output",
            "__end__": "__end__",
        })

    g.set_entry_point("load_context")
    g.add_edge("load_context", "guardrail_input")
    g.add_conditional_edges("guardrail_input",
                            lambda s: "finalize" if s.metadata.get("guardrail_blocked") else "supervisor",
                            {"supervisor": "supervisor", "finalize": "finalize"})
    g.add_conditional_edges("supervisor", router_supervisor, {
        "route_agent": "route_agent",
        "ticket_agent": "ticket_agent",
        "travel_agent": "travel_agent",
        "qa_agent": "qa_agent",
        # Plan-and-Execute：阶段推进与收尾
        "supervisor": "supervisor",
        "guardrail_output": "guardrail_output",
    })
    g.add_edge("guardrail_output", "finalize")
    return g.compile()


__all__ = [
    "SPECIALISTS", "SpecialistConfig", "build_specialist_graph",
    "build_orchestrator",
]
