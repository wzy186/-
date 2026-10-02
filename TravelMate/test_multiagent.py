"""多 Agent 冒烟测试（Mock 模式，无需任何 API Key）。

运行: ./.venv/bin/python test_multiagent.py
"""
import json
import sys

sys.path.insert(0, ".")

from core.agent import process, resume  # noqa: E402
from core.mcp_client import get_mcp_status  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")


def show(r: dict):
    print(f"  reply: {r['reply'][:120]!r}")
    print(f"  thinking: {json.dumps(r['thinking'], ensure_ascii=False)[:200]}")
    print(f"  trace nodes: {[t['node'] for t in r['trace']]}")


def called_tool(r: dict, tool: str) -> bool:
    """从 thinking 追踪中判断工具是否被调用（tool_calls 最终轮会被清空）。"""
    return any(f"调用工具 {tool}(" in step for step in r["thinking"])


print("== 1. 路线 Agent：'从新宿到浅草寺怎么走' ==")
r = process("从新宿到浅草寺怎么走")
show(r)
check("路由到 route_agent", "route_agent" in json.dumps(r["trace"]))
check("调用了 route 工具", called_tool(r, "route"))
check("有路线结果回复", len(r["reply"]) > 10)

print("\n== 2. 票务 Agent：'帮我查北京到上海的高铁票' ==")
r = process("帮我查北京到上海的高铁票")
show(r)
check("路由到 ticket_agent", "ticket_agent" in json.dumps(r["trace"]))
check("调用了 train 工具", called_tool(r, "train"))
check("回复含车次信息", "G" in r["reply"] or "车次" in r["reply"] or "12306" in r["reply"])

print("\n== 3. 票务 Agent HITL：'帮我订北京到东京最便宜的机票，乘客是张三' ==")
r = process("帮我订北京到东京最便宜的机票，乘客是张三")
show(r)
check("触发 HITL 中断", r["interrupted"])
check("挂起 book_flight", any(a["tool"] == "book_flight" for a in r.get("interrupt_data", {}).get("actions", [])))

print("  → 用户确认")
r2 = resume("default", "yes")
print(f"  reply: {r2['reply'][:150]!r}")
check("确认后执行预订", any(a["tool"] == "book_flight" for a in r2["actions"]))
check("回复含订单确认", "预订成功" in r2["reply"])

print("\n== 4. 行程 Agent：'东京天气怎么样' ==")
r = process("东京天气怎么样")
show(r)
check("路由到 travel_agent", "travel_agent" in json.dumps(r["trace"]))
check("调用了 weather 工具", called_tool(r, "weather"))

print("\n== 5. 多意图拆分：'查一下北京到上海的高铁，再看看路线怎么走' ==")
r = process("查一下北京到上海的高铁，再看看从火车站到外滩的路线怎么走")
show(r)
check("派发了多个任务", "ticket_agent" in json.dumps(r["thinking"]) and "route_agent" in json.dumps(r["thinking"]))
check("两个 Agent 都有产出", len(r["reply"]) > 20)

print("\n== 6. 问答 Agent：'东京有什么好玩的' ==")
r = process("东京有什么好玩的")
show(r)
check("路由到 qa_agent", "qa_agent" in json.dumps(r["trace"]))
check("有回答", len(r["reply"]) > 10)

print("\n== 7. MCP Server 状态（未配置时应为空）==")
status = get_mcp_status()
print(f"  {status}")
check("未配置 MCP 时无连接报错", isinstance(status, dict))

print(f"\n{'=' * 40}\n结果: {PASS} 通过, {FAIL} 失败")
sys.exit(1 if FAIL else 0)
