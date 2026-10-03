import os
import json
import httpx
from dotenv import load_dotenv
from tools.base import BaseTool

# 确保在 core.llm 之前被导入时也能读到 .env
load_dotenv()


class AmapTool(BaseTool):
    """高德地图 MCP 工具：路线规划、周边搜索、地理编码、行政区划"""

    def __init__(self, action: str = "route"):
        self.action = action
        self.key = os.getenv("AMAP_API_KEY", "")

    @property
    def name(self):
        return f"amap_{self.action}"

    @property
    def description(self):
        return {
            "route": "路线规划（驾车/公交/步行/骑行）",
            "nearby": "周边搜索（餐厅/药店/ATM等）",
            "geocode": "地理编码（地址↔坐标）",
            "district": "行政区划查询",
        }.get(self.action, "高德地图")

    def run(self, args: dict) -> str:
        if not self.key:
            return self._mock(args)
        try:
            if self.action == "route":
                return self._route(args)
            elif self.action == "nearby":
                return self._nearby(args)
            elif self.action == "geocode":
                return self._geocode(args)
            elif self.action == "district":
                return self._district(args)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e),
                               "message": f"⚠️ 高德API调用失败: {e}，请检查 AMAP_API_KEY 是否有效"}, ensure_ascii=False)
        return "未知操作"

    # ── Real API calls ──

    def _resolve_location(self, value: str, fallback_hint: str = "") -> str:
        """地名 → 坐标（"lon,lat"）。

        解析顺序：已是坐标 → 地理编码API（结构化地址）→ POI搜索API（车站/地标等）。
        """
        value = (value or "").strip()
        if not value:
            value = fallback_hint
        if not value:
            return ""
        import re
        if re.match(r"^-?\d+\.?\d*\s*,\s*-?\d+\.?\d*$", value):
            return value.replace(" ", "")
        # 1) 地理编码（地址）
        try:
            result = json.loads(self._geocode({"address": value}))
            for g in (result or {}).get("geocodes") or []:
                if g.get("location"):
                    return g["location"]
        except Exception:
            pass
        # 2) POI 关键字搜索（车站/地标/景点）
        loc = self._search_poi(value)
        if loc:
            return loc
        raise ValueError(f"无法解析地点坐标: {value}（可尝试加上城市名，如'北京市中关村'）")

    def _search_poi(self, keywords: str) -> str:
        url = "https://restapi.amap.com/v3/place/text"
        params = {"key": self.key, "keywords": keywords, "offset": 1, "page": 1}
        r = httpx.get(url, params=params, timeout=10)
        pois = (r.json() or {}).get("pois") or []
        return pois[0].get("location", "") if pois else ""

    def _route(self, args: dict) -> str:
        # 支持地名或坐标：origin_name/destination_name 优先，其次 origin/destination
        origin = self._resolve_location(
            args.get("origin_name") or args.get("origin") or "", "起点")
        destination = self._resolve_location(
            args.get("destination_name") or args.get("destination") or "", "终点")
        mode = (args.get("mode") or "驾车").strip()

        # 公交/transit 接口
        if "公交" in mode or "地铁" in mode or "transit" in mode.lower():
            url = "https://restapi.amap.com/v3/direction/transit/integrated"
            params = {"key": self.key, "origin": origin, "destination": destination,
                      "city": args.get("city", "北京"), "strategy": "0"}
            r = httpx.get(url, params=params, timeout=10)
            return self._format_route_response(r.json(), mode, origin, destination)

        # 步行 / 骑行 / 驾车
        if "步行" in mode:
            url = "https://restapi.amap.com/v3/direction/walking"
        elif "骑行" in mode or "自行车" in mode:
            url = "https://restapi.amap.com/v5/direction/bicycling"
        else:
            url = "https://restapi.amap.com/v3/direction/driving"
        params = {"key": self.key, "origin": origin, "destination": destination}
        if url.endswith("driving"):
            params["strategy"] = args.get("strategy", "0")
        r = httpx.get(url, params=params, timeout=10)
        return self._format_route_response(r.json(), mode, origin, destination)

    def _format_route_response(self, data: dict, mode: str, origin: str, destination: str) -> str:
        """把高德原始响应整理成精简结构，并附原始数据。"""
        status = data.get("status") == "1"
        if not status:
            info = data.get("info", "未知错误")
            return json.dumps({"success": False, "message": f"⚠️ 高德路线查询失败: {info}",
                               "origin": origin, "destination": destination}, ensure_ascii=False)
        route = data.get("route", {})
        out = {"success": True, "mode": mode, "origin": origin, "destination": destination}
        if mode == "公交":
            transits = []
            for t in (route.get("transits") or [])[:3]:
                segs = []
                for seg in (t.get("segments") or []):
                    bus = (seg.get("bus") or {}).get("buslines") or []
                    for b in bus[:2]:
                        segs.append(f"{b.get('name','')}（{(b.get('departure_stop') or {}).get('name','')}→{(b.get('arrival_stop') or {}).get('name','')}）")
                    walk = seg.get("walking")
                    if walk and walk.get("distance") not in (None, "0"):
                        segs.append(f"步行{walk.get('distance')}米")
                transits.append({"duration_min": round(int(t.get("duration", 0)) / 60),
                                 "walking_m": t.get("walking_distance", "0"),
                                 "price": t.get("cost", ""),
                                 "segments": segs})
            out["transits"] = transits
            out["message"] = f"高德公交路线：{origin}→{destination}，共{len(transits)}个方案" if transits else "未找到公交方案"
        else:
            paths = route.get("paths") or []
            p0 = paths[0] if paths else {}
            steps = [s.get("instruction", "") for s in (p0.get("steps") or [])[:12]]
            out["distance_km"] = round(int(p0.get("distance", 0)) / 1000, 1)
            out["duration_min"] = round(int(p0.get("duration", 0)) / 60)
            out["steps"] = steps
            out["message"] = (f"高德{mode}路线：{origin}→{destination}，"
                              f"全程{out['distance_km']}km，约{out['duration_min']}分钟")
        out["raw"] = data
        return json.dumps(out, ensure_ascii=False)

    def _nearby(self, args: dict) -> str:
        location = self._resolve_location(
            args.get("location_name") or args.get("location") or "116.397428,39.90923")
        keywords = args.get("keywords", "餐厅")
        radius = args.get("radius", 3000)
        url = "https://restapi.amap.com/v3/place/around"
        params = {"key": self.key, "location": location, "keywords": keywords,
                  "radius": radius, "offset": 5}
        r = httpx.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") != "1":
            return json.dumps({"success": False, "message": f"⚠️ 高德周边搜索失败: {data.get('info','')}",
                               "keywords": keywords}, ensure_ascii=False)
        pois = [{"name": p.get("name"), "address": p.get("address"),
                 "distance_m": p.get("distance"), "type": p.get("type")}
                for p in (data.get("pois") or [])[:8]]
        return json.dumps({"success": True, "keywords": keywords, "radius": radius,
                           "count": len(pois), "places": pois,
                           "message": f"高德周边搜索「{keywords}」：共{len(pois)}个结果"},
                          ensure_ascii=False)

    def _geocode(self, args: dict) -> str:
        address = args.get("address", "")
        url = "https://restapi.amap.com/v3/geocode/geo"
        params = {"key": self.key, "address": address}
        r = httpx.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") != "1" or not data.get("geocodes"):
            return json.dumps({"success": False, "message": f"⚠️ 地理编码失败: {data.get('info','')}（{address}）"},
                              ensure_ascii=False)
        g = data["geocodes"][0]
        return json.dumps({"success": True, "address": address,
                           "location": g.get("location"), "province": g.get("province"),
                           "city": g.get("city"), "district": g.get("district"),
                           "formatted": g.get("formatted_address"),
                           "message": f"地理编码：{g.get('formatted_address', address)} → {g.get('location')}",
                           "raw": data}, ensure_ascii=False)

    def _district(self, args: dict) -> str:
        keywords = args.get("keywords", "")
        url = "https://restapi.amap.com/v3/config/district"
        params = {"key": self.key, "keywords": keywords, "subdistrict": 1}
        r = httpx.get(url, params=params, timeout=10)
        return r.text

    # ── Rich mock data ──

    def _mock(self, args: dict) -> str:
        if self.action == "route":
            return self._mock_route(args)
        elif self.action == "nearby":
            return self._mock_nearby(args)
        elif self.action == "geocode":
            return self._mock_geocode(args)
        elif self.action == "district":
            return self._mock_district(args)
        return "高德地图模拟数据"

    def _mock_route(self, args):
        origin = args.get("origin_name", args.get("origin", "新宿站"))
        dest = args.get("destination_name", args.get("destination", "浅草寺"))
        mode = args.get("mode", "驾车")

        routes = {
            "驾车": {"distance": "12.5km", "duration": "35分钟", "tolls": "0日元", "fuel_cost": "约800日元",
                "steps": [f"从{origin}出发", "沿明治通向东行驶2.3km", "右转进入中央通", "经过秋叶原站", "左转进入浅草通", f"到达{dest}"],
                "warnings": ["新宿周边停车费每小时600-1200日元", "工作日早高峰7-9点严重拥堵"]},
            "公交": {"distance": "10.8km", "duration": "28分钟", "cost": "250日元",
                "lines": ["JR中央线(新宿→神田) 8分钟", "地铁银座线(神田→浅草) 12分钟", "步行至浅草寺 5分钟"],
                "transfer": "神田站换乘（步行2分钟）", "first_train": "05:05", "last_train": "00:03"},
            "步行": {"distance": "9.2km", "duration": "1小时50分钟", "calories": "约350kcal",
                "route": [f"从{origin}出发", "沿靖国通向东", "经过神保町古书街", "沿中央通继续", "经过秋叶原", f"到达{dest}"],
                "highlights": ["途经秋叶原电器街", "神保町世界最大古书街"]},
            "骑行": {"distance": "10.1km", "duration": "45分钟", "rental": "Docomo自行车 150日元/30分钟",
                "route": [f"从{origin}出发", "沿自行车道东行", "经过皇居北之丸公园", f"到达{dest}"],
                "tips": "东京部分区域禁止骑车，注意标识"},
        }

        data = routes.get(mode, routes["驾车"])
        result = {
            "origin": origin, "destination": dest, "mode": mode,
            "data": data,
            "tip": "东京市内建议地铁出行，驾车停车费高(每小时600-1200日元)",
            "transit_alternative": {"mode": "公交", "duration": "28分钟", "cost": "250日元", "lines": ["JR中央线→地铁银座线"]},
        }
        return json.dumps(result, ensure_ascii=False)

    def _mock_nearby(self, args):
        location = args.get("location_name", args.get("location", "新宿站"))
        kw = args.get("keywords", "餐厅")
        radius = args.get("radius", 3000)

        nearby_db = {
            "餐厅": [
                {"name": "和食処 つじ半 新宿店", "address": "新宿3-15-15 B1F", "distance": "150m", "rating": "4.3", "tel": "03-3352-1234", "price": "¥2000-4000", "hours": "11:00-23:00", "type": "日式料理", "tip": "午餐套餐性价比高"},
                {"name": "一蘭拉面 新宿中央店", "address": "新宿3-34-6", "distance": "280m", "rating": "4.1", "tel": "03-3225-1551", "price": "¥1000-1500", "hours": "24小时", "type": "拉面", "tip": "深夜也排队，单人隔间设计"},
                {"name": "鳥貴族 新宿西口店", "address": "新宿西口1-12-5", "distance": "350m", "rating": "4.0", "tel": "03-3345-6789", "price": "¥2000-3000", "hours": "17:00-24:00", "type": "居酒屋", "tip": "所有串烤280日元(含税)，便宜"},
                {"name": "築地銀だこ 新宿南口店", "address": "新宿南口2-5-1", "distance": "200m", "rating": "3.9", "tel": "03-3341-0091", "price": "¥500-800", "hours": "10:00-22:00", "type": "章鱼烧", "tip": "现做热乎章鱼烧，外脆内软"},
            ],
            "便利店": [
                {"name": "7-Eleven 新宿三丁目店", "address": "新宿3-5-2", "distance": "50m", "rating": "4.2", "tel": "03-3352-7711", "hours": "24小时", "type": "便利店", "tip": "ATM取现手续费最低110日元"},
                {"name": "FamilyMart 新宿东口店", "address": "新宿3-17-5", "distance": "120m", "rating": "4.0", "tel": "03-3341-2233", "hours": "24小时", "type": "便利店", "tip": "FamiPort可打印文档"},
                {"name": "Lawson 新宿站前店", "address": "新宿西口1-1-3", "distance": "180m", "rating": "4.0", "tel": "03-3345-4455", "hours": "24小时", "type": "便利店", "tip": "Lawson Select自有品牌品质高"},
            ],
            "药妆店": [
                {"name": "松本清 新宿东口店", "address": "新宿3-23-6", "distance": "200m", "rating": "4.0", "tel": "03-3341-0091", "price": "药妆", "hours": "10:00-23:00", "type": "药妆店", "tip": "出示优惠券享95折"},
                {"name": "大国药妆 新宿店", "address": "新宿3-28-12", "distance": "350m", "rating": "3.9", "tel": "03-3356-7890", "price": "药妆", "hours": "09:00-24:00", "type": "药妆店", "tip": "价格通常比松本清便宜5%"},
            ],
        }

        places = nearby_db.get(kw, [
            {"name": f"{kw}·{location}店", "address": f"{location}附近", "distance": "200m", "rating": "4.0", "tel": "03-1234-5678", "type": kw},
            {"name": f"{kw}·{location}站前店", "address": f"{location}站前", "distance": "350m", "rating": "3.9", "tel": "03-2345-6789", "type": kw},
        ])

        return json.dumps({
            "location": location, "keywords": kw, "radius": radius,
            "places": places, "count": len(places),
            "tip": f"搜索半径{radius}米内共{len(places)}个结果",
        }, ensure_ascii=False)

    def _mock_geocode(self, args):
        address = args.get("address", "东京塔")
        geocode_db = {
            "东京塔": {"location": "139.7454,35.6586", "province": "东京都", "city": "东京", "district": "港区", "street": "芝公园"},
            "浅草寺": {"location": "139.7968,35.7148", "province": "东京都", "city": "东京", "district": "台东区", "street": "浅草"},
            "新宿站": {"location": "139.7005,35.6897", "province": "东京都", "city": "东京", "district": "新宿区", "street": "新宿3丁目"},
            "涩谷站": {"location": "139.7012,35.6580", "province": "东京都", "city": "东京", "district": "涩谷区", "street": "涩谷2丁目"},
            "埃菲尔铁塔": {"location": "2.2945,48.8584", "province": "巴黎", "city": "巴黎", "district": "7区", "street": "战神广场"},
            "卢浮宫": {"location": "2.3376,48.8606", "province": "巴黎", "city": "巴黎", "district": "1区", "street": "里沃利街"},
        }
        data = geocode_db.get(address, {"location": "139.6917,35.6895", "province": "未知", "city": address, "district": "未知", "street": ""})
        result = {"address": address, **data, "formatted_address": f"{data.get('province', '')}{data.get('city', '')}{data.get('district', '')}{data.get('street', '')}"}
        return json.dumps(result, ensure_ascii=False)

    def _mock_district(self, args):
        kw = args.get("keywords", "东京")
        district_db = {
            "东京": {"name": "东京都", "level": "都道府县", "center": "139.6917,35.6895", "districts": ["新宿区", "涩谷区", "港区", "千代田区", "中央区", "台东区", "墨田区", "江东区", "品川区", "目黑区", "世田谷区", "丰岛区", "练马区", "板桥区", "杉並区", "北区", "荒川区", "足立区", "葛饰区", "江户川区", "大田区", "中野区", "文京区"], "population": "1396万人", "area": "2194km²"},
            "巴黎": {"name": "巴黎", "level": "城市", "center": "2.3522,48.8566", "districts": ["1区(Louvre)", "2区(Bourse)", "3区(Temple)", "4区(Hôtel-de-Ville)", "5区(Panthéon)", "6区(Luxembourg)", "7区(Palais-Bourbon)", "8区(Élysée)", "9区(Opéra)", "10区(Entrepôt)", "11区(Popincourt)", "12区(Reuilly)", "13区(Gobelins)", "14区(Observatoire)", "15区(Vaugirard)", "16区(Passy)", "17区(Batignolles)", "18区(Butte-Montmartre)", "19区(Buttes-Chaumont)", "20区(Ménilmontant)"], "population": "215万人", "area": "105km²"},
            "曼谷": {"name": "曼谷", "level": "城市", "center": "100.5018,13.7563", "districts": ["是隆区", "Siam区", "考山区", "素坤逸区", "河滨区", "唐人街", "RCA区"], "population": "1057万人", "area": "1569km²"},
        }
        data = district_db.get(kw, {"name": kw, "level": "城市", "center": "0,0", "districts": ["市中心"], "population": "未知", "area": "未知"})
        return json.dumps({"keywords": kw, **data}, ensure_ascii=False)
