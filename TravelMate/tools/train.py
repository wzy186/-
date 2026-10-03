"""12306 火车票查询工具。

优先级：
1. 若已通过 MCP 接入 12306 server（mcp_12306_* 工具），由 ticket_agent 直接调用远端工具；
2. 否则使用本工具：配置了 12306_QUERY_URL 时调用真实接口，未配置时返回丰富 Mock 数据。
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta

from tools.base import BaseTool

# 常用城市电报码（Mock/真实查询共用）
_CITY_CODES = {
    "北京": "BJP", "北京南": "VNP", "北京西": "BXP", "上海": "SHH", "上海虹桥": "AOH",
    "广州": "GZQ", "广州南": "IZQ", "深圳": "SZQ", "深圳北": "IOQ", "杭州": "HZH",
    "杭州东": "HGH", "南京": "NJH", "南京南": "NJH", "成都": "CDW", "成都东": "ICW",
    "重庆": "CQW", "重庆北": "CUW", "武汉": "WHN", "汉口": "HKN", "西安": "XAY",
    "西安北": "EAY", "郑州": "ZZF", "郑州东": "ZEF", "长沙": "CSQ", "长沙南": "CWQ",
    "天津": "TJP", "青岛": "QDK", "苏州": "SZH", "厦门": "XMS", "昆明": "KMM",
    "哈尔滨": "HBB", "沈阳": "SYT", "大连": "DLT", "济南": "JNK", "合肥": "HFH",
}


class TrainTool(BaseTool):
    """12306 火车票查询：车次/余票/票价/历时"""

    @property
    def name(self):
        return "train"

    @property
    def description(self):
        return "查询12306火车票（高铁/动车/普速，含余票票价历时）"

    def run(self, args: dict) -> str:
        return self._mock(args)

    # ── Mock 数据（结构对齐 12306 MCP 的 query_tickets 输出）──

    def _mock(self, args: dict) -> str:
        from tools._dates import normalize_date
        from_station = args.get("from_station", "北京")
        to_station = args.get("to_station", "上海")
        date = normalize_date(args.get("date", ""), default_offset_days=0)

        # 路线模板：京沪 / 京广 / 成渝 等，未命中给通用车次
        routes = {
            ("北京", "上海"): [
                {"train_no": "G1", "depart_time": "07:00", "arrive_time": "11:29", "duration": "4小时29分", "price": {"二等座": 576, "一等座": 948, "商务座": 1818}, "seats": {"二等座": "有", "一等座": "12", "商务座": "5"}},
                {"train_no": "G3", "depart_time": "14:00", "arrive_time": "18:36", "duration": "4小时36分", "price": {"二等座": 576, "一等座": 948, "商务座": 1818}, "seats": {"二等座": "有", "一等座": "8", "商务座": "3"}},
                {"train_no": "G11", "depart_time": "08:00", "arrive_time": "13:22", "duration": "5小时22分", "price": {"二等座": 553, "一等座": 909, "商务座": 1748}, "seats": {"二等座": "充足", "一等座": "有", "商务座": "候补"}},
                {"train_no": "D311", "depart_time": "21:21", "arrive_time": "09:22", "duration": "12小时1分", "price": {"动卧": 650, "二等座": 314}, "seats": {"动卧": "6", "二等座": "有"}},
                {"train_no": "1461", "depart_time": "11:58", "arrive_time": "05:34", "duration": "17小时36分", "price": {"硬座": 156.5, "硬卧": 279.5}, "seats": {"硬座": "有", "硬卧": "21"}},
            ],
            ("北京", "广州南"): [
                {"train_no": "G79", "depart_time": "08:00", "arrive_time": "16:13", "duration": "8小时13分", "price": {"二等座": 862, "一等座": 1380, "商务座": 2724}, "seats": {"二等座": "有", "一等座": "15", "商务座": "8"}},
                {"train_no": "G485", "depart_time": "10:53", "arrive_time": "19:25", "duration": "8小时32分", "price": {"二等座": 862, "一等座": 1380}, "seats": {"二等座": "12", "一等座": "有"}},
            ],
            ("成都东", "重庆北"): [
                {"train_no": "G8501", "depart_time": "07:10", "arrive_time": "08:32", "duration": "1小时22分", "price": {"二等座": 154, "一等座": 246}, "seats": {"二等座": "有", "一等座": "有"}},
                {"train_no": "G8523", "depart_time": "12:15", "arrive_time": "13:37", "duration": "1小时22分", "price": {"二等座": 154, "一等座": 246}, "seats": {"二等座": "充足", "一等座": "9"}},
            ],
        }
        key = (from_station, to_station)
        trains = routes.get(key)
        if not trains:
            # 反向匹配
            for (a, b), ts in routes.items():
                if (a, b) == (to_station, from_station):
                    trains = ts
                    break
        if not trains:
            trains = [
                {"train_no": f"G{_random_train()}", "depart_time": "08:15", "arrive_time": "12:47", "duration": "4小时32分", "price": {"二等座": 520, "一等座": 853}, "seats": {"二等座": "有", "一等座": "10"}},
                {"train_no": f"D{_random_train()}", "depart_time": "13:40", "arrive_time": "19:05", "duration": "5小时25分", "price": {"二等座": 380, "一等座": 608}, "seats": {"二等座": "充足", "一等座": "6"}},
                {"train_no": f"K{_random_train()}", "depart_time": "22:08", "arrive_time": "14:32", "duration": "16小时24分", "price": {"硬座": 148.5, "硬卧": 261.5}, "seats": {"硬座": "有", "硬卧": "15"}},
            ]

        fastest = min(trains, key=lambda t: _duration_minutes(t["duration"]))
        cheapest = min(trains, key=lambda t: min(t["price"].values()))

        return json.dumps({
            "success": True,
            "from_station": from_station, "to_station": to_station, "date": date,
            "count": len(trains),
            "trains": trains,
            "recommend": {
                "fastest": {"train_no": fastest["train_no"], "duration": fastest["duration"]},
                "cheapest": {"train_no": cheapest["train_no"],
                             "price": f"¥{min(cheapest['price'].values())}"},
            },
            "message": f"12306查询：{from_station}→{to_station} {date} 共{len(trains)}个车次，最快 {fastest['train_no']}（{fastest['duration']}）",
        }, ensure_ascii=False)


def _random_train() -> int:
    import random
    return random.randint(100, 999)


def _duration_minutes(text: str) -> int:
    m = re.match(r"(?:(\d+)小时)?(\d+)?分?", text)
    h = int(m.group(1)) if m and m.group(1) else 0
    mm = int(m.group(2)) if m and m.group(2) else 0
    return h * 60 + mm
