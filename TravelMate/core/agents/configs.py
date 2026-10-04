"""四个专家子 Agent 的配置：路线规划（高德）/ 票务（12306+航班）/ 行程规划 / 知识问答。"""

from __future__ import annotations

import json

from core.agents.base_nodes import SpecialistConfig
from core.memory import get_profile

# ────────────────────────── 工具结果 → 文本摘要（Mock 收尾用）──────────────────────────


def _summarize_results(state) -> str:
    parts = []
    for tr in state.tool_results:
        data = tr.get("result", {})
        if isinstance(data, dict) and data.get("message"):
            parts.append(data["message"])
        elif isinstance(data, dict) and data:
            # 无 message 字段时输出紧凑摘要
            parts.append(json.dumps(data, ensure_ascii=False)[:400])
    return "\n\n".join(parts) if parts else ""


# ────────────────────────── 1. 路线规划 Agent（高德）──────────────────────────

ROUTE_AGENT_PROMPT = """你是 TravelMate 的路线规划专家（路线Agent），专注地图与出行导航：
- 路线规划：驾车/公交/步行/骑行，给出距离、耗时、换乘方案、费用
- 周边搜索：目的地附近的餐厅/便利店/药店/充电站等
- 地理编码与行政区划：地址→坐标、城市区域信息

## 强制规则（最重要）
- 只要用户询问具体路线、距离、耗时、怎么走，**必须调用 route 工具**获取真实路网数据，禁止凭记忆回答
- route 的 mode 参数只接受：驾车 / 公交 / 步行 / 骑行（中文）
- 查询附近设施**必须调用 nearby 工具**
- 拿到工具结果后，把距离/耗时/换乘段/费用整理成结构化回答，不要原样堆 JSON
- 若工具调用失败，明确告知用户并给出常识性建议，同时说明数据可能不准确
- 若用户问题与路线/地图完全无关，简短说明并建议咨询其他模块"""


def _route_mock(state, instruction: str, iters: int) -> str:
    if iters > 0:
        return _summarize_results(state) or "已为您完成路线查询。"
    text = instruction.lower()
    if any(k in text for k in ["附近", "周边", "餐厅", "便利店", "药店", "厕所", "加油站"]):
        kw = "餐厅"
        for k in ["便利店", "药店", "加油站", "餐厅"]:
            if k in text:
                kw = k
                break
        return f'[call:nearby] {{"location_name":"目的地","keywords":"{kw}","radius":3000}}'
    import re
    if any(k in text for k in ["路线", "怎么走", "怎么去", "导航", "多远", "驾车", "公交", "地铁", "步行", "骑行"]) \
            or re.search(r"怎么(过去|走|去|到|前往|到达)|如何(去|前往|到达)", instruction):
        mode = "驾车"
        for m in [("公交", "公交"), ("地铁", "公交"), ("步行", "步行"), ("骑行", "骑行"), ("驾车", "驾车")]:
            if m[0] in text:
                mode = m[1]
                break
        # 提取 "从A到B" 的起终点
        import re
        origin, dest = "出发点", "目的地"
        m = re.search(r"从(.{1,12}?)到(.{1,12}?)[，,。\s]|从(.{1,12}?)到(.{1,12})$", instruction)
        if m:
            origin = (m.group(1) or m.group(3) or origin).strip()
            dest = (m.group(2) or m.group(4) or dest).strip()
        return f'[call:route] {{"origin_name":"{origin}","destination_name":"{dest}","mode":"{mode}"}}'
    return '[call:geocode] {"address":"目的地"}'


# ────────────────────────── 2. 票务 Agent（12306 MCP + 航班）──────────────────────────

TICKET_AGENT_PROMPT = """你是 TravelMate 的票务专家（票务Agent），负责火车票与机票：
- 火车票：通过 train 工具查询 12306 余票/票价/历时（若 12306 MCP 已接入会自动提供 mcp_12306_* 工具，优先使用）
- 机票：通过 flight 工具查询多航司比价
- 预订：采用**推荐制**——查询后直接推荐最优车次/航班（优先有票且最快，其次最便宜），
  立即调用 book_train_ticket / book_flight 发起预订，让用户在确认环节定夺
- **禁止询问乘客姓名/座位**：不要输出"请提供乘客姓名""请确认以下信息，我立即为您预订"这类
  追问话术。乘客未提供时默认使用用户画像/登录名（不要虚构"张三"等示例名）。预订由你直接发起，措辞用
  "已为您推荐并锁定 G3 一等座，请在下方确认框中核对信息"，确认与否由用户点按钮决定
- 确认后按工具返回的消息输出预订结果。

回答要求：以表格或列表呈现可选车次/航班，标注推荐项（最快/最便宜）。"""


def _ticket_mock(state, instruction: str, iters: int) -> str:
    if iters > 0:
        text = _summarize_results(state)
        # 有预订意图 → 直接推荐最优车次发起预订（HITL 确认兜底）
        if any(k in instruction.lower() for k in ["订", "买", "book"]) and state.tool_results:
            trains = []
            for tr in state.tool_results:
                data = tr.get("result", {})
                trains = data.get("trains") or []
                if trains:
                    break
            if trains:
                # 推荐优先级：二等座有票 → 最快
                pick = None
                for t in trains:
                    seats = t.get("seats", {})
                    if "二等座" in seats and seats["二等座"] in ("有", "充足"):
                        pick = t
                        break
                pick = pick or trains[0]
                profile = get_profile()
                passenger = (profile or {}).get("name") or "待补充"
                price = pick.get("price", {}).get("二等座", 0)
                return (f'[call:book_train_ticket] {{"train_no":"{pick["train_no"]}",'
                        f'"from_station":"{state.tool_results[0]["result"].get("from_station", "出发站")}",'
                        f'"to_station":"{state.tool_results[0]["result"].get("to_station", "到达站")}",'
                        f'"date":"{state.tool_results[0]["result"].get("date", "")}",'
                        f'"seat_type":"二等座","passenger":"{passenger}","price":{price},'
                        f'"depart_time":"{pick.get("depart_time", "")}","arrive_time":"{pick.get("arrive_time", "")}"}}'
                        f"\n\n已为您选定推荐车次 **{pick['train_no']}**（二等座 ¥{price}），请在确认框中核对信息。")
        return text or "已完成票务查询。"
    text = instruction.lower()
    # 提取城市与日期
    cities = ["北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "重庆", "武汉", "西安",
              "郑州", "长沙", "天津", "青岛", "苏州", "厦门", "昆明", "哈尔滨", "沈阳", "大连",
              "东京", "巴黎", "曼谷", "首尔", "伦敦", "纽约", "悉尼", "迪拜", "罗马", "巴厘岛"]
    found = [c for c in cities if c in instruction]
    date = ""
    import re
    m = re.search(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}月\d{1,2}[号日]?)", instruction)
    if m:
        date = m.group(1)
    train_like = any(k in text for k in ["火车", "高铁", "动车", "12306", "车票", "余票", "订火车"])
    flight_like = any(k in text for k in ["机票", "航班", "飞", "订飞机"])
    if not train_like and not flight_like:
        train_like = "订票" in text or "买票" in text
        flight_like = not train_like
    if train_like and len(found) >= 2:
        return (f'[call:train] {{"from_station":"{found[0]}","to_station":"{found[1]}","date":"{date}"}}'
                f"\n\n正在为您查询 {found[0]}→{found[1]} 的车次…")
    if flight_like and len(found) >= 2:
        # 明确的预订意图 + 已知乘客 → 查询后直接发起预订（走 HITL 确认）
        wants_book = any(k in text for k in ["订", "预订", "买", "book"])
        if wants_book:
            passenger = ""
            m = re.search(r"乘客[是为：:\s]*(\S+)", instruction)
            if m:
                passenger = m.group(1).rstrip("。，,")
            else:
                profile = get_profile()
                passenger = (profile or {}).get("name", "")
            if passenger:
                return (f'[call:flight] {{"departure":"{found[0]}","destination":"{found[1]}","date":"{date}"}}'
                        f"\n\n根据查询结果，为您预订最便宜的航班："
                        f"\n\n[call:book_flight] {{\"airline\":\"春秋 9C8515\",\"departure\":\"{found[0]}\","
                        f"\"arrival\":\"{found[1]}\",\"date\":\"{date}\",\"passenger\":\"{passenger}\","
                        f"\"price\":1500,\"seat\":\"无偏好\",\"meal\":\"标准\"}}")
        return (f'[call:flight] {{"departure":"{found[0]}","destination":"{found[1]}","date":"{date}"}}'
                f"\n\n正在为您查询 {found[0]}→{found[1]} 的航班…")
    if train_like or flight_like:
        tool = "train" if train_like else "flight"
        return f'[call:{tool}] {{"from_station":"北京","to_station":"上海","date":"{date}"}}' if train_like \
            else f'[call:{tool}] {{"departure":"北京","destination":"上海","date":"{date}"}}'
    return "请告诉我出发地、目的地和日期，我来帮您查询车次或航班。"


# ────────────────────────── 3. 行程规划 Agent ──────────────────────────

TRAVEL_AGENT_PROMPT = """你是 TravelMate 的行程规划专家（行程Agent），负责旅行全流程服务：
- 行程规划：按目的地/天数/预算/风格生成逐日行程（景点编排就近、含交通与餐食推荐）
- 酒店/天气/景点/预算/汇率/翻译：调用对应工具查询后综合建议
- 操作：book_hotel 预订酒店、add_spot 加行程、save_phrase 收藏短语、add_reminder 设提醒、set_note 记备注
- 参考用户偏好画像个性化推荐；结果具体（价格/时段/评分）。

回答要求：行程按 Day 1/Day 2… 结构化输出，含每日主题、景点、交通、餐食、费用小计。"""


def _travel_mock(state, instruction: str, iters: int) -> str:
    if iters > 0:
        return _summarize_results(state) or "已完成行程服务。"
    text = instruction.lower()
    if "汇率" in text or "换算" in text:
        return '[call:exchange] {"amount": 1000, "from": "CNY", "to": "JPY"}'
    if "翻译" in text:
        return '[call:translate] {"text": "谢谢", "target": "ja"}'
    if "预算" in text:
        import re
        m = re.search(r"(\d+)", instruction)
        budget = m.group(1) if m else "15000"
        return f'[call:budget] {{"budget": {budget}, "days": 5, "destination": "目的地"}}'
    if "天气" in text:
        import re
        m = re.search(r"(东京|巴黎|曼谷|首尔|伦敦|纽约|悉尼|迪拜|罗马|巴厘岛|北京|上海)", instruction)
        city = m.group(1) if m else "东京"
        return f'[call:weather] {{"destination":"{city}","days":7}}'
    if "酒店" in text and ("订" in text or "推荐" in text or "预订" in text):
        guest = get_profile().get("name", "张三") if get_profile() else "张三"
        return (f'[call:hotel] {{"destination":"目的地","budget_per_night":800,"style":"舒适"}}'
                f"\n\n[call:book_hotel] {{\"name\":\"推荐酒店\",\"city\":\"目的地\",\"check_in\":\"待确认\","
                f"\"check_out\":\"待确认\",\"guest\":\"{guest}\",\"room_type\":\"标准间\",\"price_per_night\":600,"
                f"\"nights\":1,\"guests\":1}}")
    if "加入行程" in text or "添加景点" in text or ("加" in text and "行程" in text):
        import re
        m = re.search(r"把(.+?)加入", instruction)
        spot = m.group(1) if m else "景点"
        return f'[call:add_spot] {{"name":"{spot}","city":"","note":"用户添加"}}'
    if "提醒" in text or "别忘了" in text:
        content = instruction.replace("提醒我", "").replace("别忘了", "").strip()[:50]
        return f'[call:add_reminder] {{"text":"{content}","date":"","type":"旅行提醒"}}'
    # 默认：行程规划
    return "好的，我按标准行程为您安排（Mock 模式）。如需查询天气/酒店/预算，请明确告诉我。"


# ────────────────────────── 4. 问答 Agent（RAG）─────────────────────────────────

QA_AGENT_PROMPT = """你是 TravelMate 的目的地知识专家（问答Agent）：
- 回答签证、交通、美食、安全、文化习俗、紧急求助等目的地知识问题
- 优先使用检索到的知识库上下文（标注在 system prompt 中）；没有依据时给出常识性建议并说明
- 回答结构化、实用，涉及紧急情况时优先给求助电话与短语

## 重要：你没有实时数据工具
你无法查询天气/余票/路线/价格等实时信息。当用户提出这类请求时：
- 不要道歉后干等用户给数据，调用 [call:handoff] {"agent": "travel_agent", "instruction": "<用户的具体请求>"} 移交给行程Agent（它有天气等实时工具）
- 天气/酒店/预算 → travel_agent；路线/导航 → route_agent；火车/机票 → ticket_agent
- 移交后简单说明"已为您转接相应专家"即可。只回答纯知识类问题。"""


def _qa_mock(state, instruction: str, iters: int) -> str:
    rag = state.metadata.get("rag_context", "")
    if rag:
        snippet = rag.strip()[:800]
        return f"根据知识库检索结果：\n\n{snippet}\n\n（Mock 模式：以上为 RAG 检索片段摘要。配置 LLM_API_KEY 可获得完整智能问答。）"
    return ("我是问答Agent（Mock 模式）。我可以回答目的地签证、交通、美食、安全等问题；"
            "配置 LLM_API_KEY 后将获得基于 RAG 知识库的完整智能回答。")


# ────────────────────────── 汇总 ──────────────────────────



def _ticket_ensure_action(state) -> bool:
    """预订意图明确但 LLM 未发起预订时，从查询结果自动构造预订动作（保证确认框出现）。"""
    task = state.metadata.get("current_task") or {}
    text = (task.get("instruction") or state.user_input or "")
    if not any(k in text for k in ("订", "买票", "book", "购票")):
        return False
    if state.pending_actions or state.actions:
        return False
    # 从最近的查询结果中取车次（本地 JSON 格式 / 12306 MCP 文本格式）
    import re as _re
    # 文本行示例：G531 北京西(telecode:VNP) -> 深圳北(telecode:IOQ) 08:00 -> 16:00 历时：8小时
    _ROW = _re.compile(r"([GDCK]\d{1,4})\s+([^\s|(]+)(?:\([^)]*\))?\s*->\s*([^\s|(]+)(?:\([^)]*\))?\s*"
                       r"(\d{1,2}:\d{2})\s*->\s*(\d{1,2}:\d{2})")

    for tr in reversed(state.tool_results):
        data = tr.get("result") or {}
        trains = data.get("trains") or []
        if not trains:
            # 12306 MCP 返回文本表格 → 逐行解析：车次/出发站/到达站/时刻/价格
            raw = str(data.get("raw") or data.get("message") or "")
            dm = _re.search(r"(\d{4}-\d{2}-\d{2})", raw)
            trains = []
            for m in _ROW.finditer(raw):
                seg = raw[m.end():m.end() + 260]
                prices = {}
                for seat_name in ("商务座", "一等座", "二等座", "硬卧", "硬座"):
                    pm = _re.search(seat_name + r"[^0-9¥]{0,8}¥?(\d+(?:\.\d+)?)", seg)
                    if pm:
                        prices[seat_name] = float(pm.group(1))
                trains.append({
                    "train_no": m.group(1), "depart_time": m.group(4), "arrive_time": m.group(5),
                    "price": prices or {"二等座": 0}, "seats": {},
                    "_from": m.group(2), "_to": m.group(3),
                })
            if trains:
                data = {**data,
                        "from_station": trains[0].get("_from", data.get("from_station", "")),
                        "to_station": trains[0].get("_to", data.get("to_station", "")),
                        "date": data.get("date") or (dm.group(1) if dm else "")}
        if not trains:
            continue
        pick, seat = None, ""
        for t in trains:
            seats = t.get("seats", {})
            available = [s for s, n in seats.items() if n in ("有", "充足")]
            if available:
                pick, seat = t, ("二等座" if "二等座" in available else available[0])
                break
        if not pick:
            pick, seat = trains[0], next(iter(trains[0].get("price", {})), "二等座")
        # 乘客缺省：画像姓名 → 登录用户名 → 待补充
        try:
            from utils.storage import get_current_user
            from core.memory import get_profile
            passenger = ((get_profile() or {}).get("name") or get_current_user() or "待补充")
        except Exception:
            passenger = "待补充"
        price = pick.get("price", {}).get(seat, 0)
        args = {
            "train_no": pick.get("train_no", ""),
            "from_station": data.get("from_station", ""), "to_station": data.get("to_station", ""),
            "date": data.get("date", ""), "seat_type": seat,
            "passenger": passenger, "price": price,
            "depart_time": pick.get("depart_time", ""), "arrive_time": pick.get("arrive_time", ""),
        }
        state.pending_actions = [{"tool": "book_train_ticket", "args": args, "agent": "ticket_agent"}]
        state.needs_confirmation = True
        state.interrupted = True
        price_txt = f"¥{price:g}" if isinstance(price, (int, float)) and price else "以12306实际为准"
        state.interrupt_data = {
            "type": "action_confirmation", "actions": state.pending_actions,
            "message": (
                "⚠️ 请核对高铁票预订信息：\n"
                f"• 车次：{args['train_no']}\n"
                f"• 区间：{args['from_station']} → {args['to_station']}\n"
                f"• 日期：{args['date'] or '（以12306为准）'}\n"
                f"• 时间：{args['depart_time']} 出发 → {args['arrive_time']} 到达\n"
                f"• 席别：{seat}　票价：{price_txt}\n"
                f"• 乘客：{passenger}\n"
                "确认后即完成预订（模拟订单，无真实扣款）。"
            ),
        }
        state.metadata["resume_node"] = "guardrail_output"
        state.thinking.append(f"[票务Agent] 自动构造预订请求 {args['train_no']}（等待用户确认）")
        return True
    return False


SPECIALISTS: dict[str, SpecialistConfig] = {
    "route_agent": SpecialistConfig(
        name="route_agent", label="路线规划Agent",
        system_prompt=ROUTE_AGENT_PROMPT,
        tools={"route", "nearby", "geocode", "district"},
        remote_prefixes=("mcp_amap",),
        mock=_route_mock, mock_intent="route",
    ),
    "ticket_agent": SpecialistConfig(
        name="ticket_agent", label="票务Agent",
        system_prompt=TICKET_AGENT_PROMPT,
        tools={"train", "flight", "book_train_ticket", "book_flight"},
        remote_prefixes=("mcp_12306",),
        mock=_ticket_mock, mock_intent="ticket",
        ensure_action=_ticket_ensure_action,
    ),
    "travel_agent": SpecialistConfig(
        name="travel_agent", label="行程规划Agent",
        system_prompt=TRAVEL_AGENT_PROMPT,
        tools={"weather", "hotel", "attraction", "budget", "exchange", "translate",
               "book_hotel", "add_spot", "save_phrase", "add_reminder", "set_note"},
        mock=_travel_mock, mock_intent="",
    ),
    "qa_agent": SpecialistConfig(
        name="qa_agent", label="问答Agent",
        system_prompt=QA_AGENT_PROMPT,
        tools=set(),
        mock=_qa_mock, mock_intent="",
    ),
}
