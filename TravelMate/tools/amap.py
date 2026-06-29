import os
import json
import httpx
from tools.base import BaseTool


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
            return json.dumps({"error": str(e), "fallback": "使用模拟数据"}, ensure_ascii=False)
        return "未知操作"

    # ── Real API calls ──

    def _route(self, args: dict) -> str:
        origin = args.get("origin", "")
        destination = args.get("destination", "")
        strategy = args.get("strategy", "0")
        url = "https://restapi.amap.com/v3/direction/driving"
        params = {"key": self.key, "origin": origin, "destination": destination, "strategy": strategy}
        r = httpx.get(url, params=params, timeout=10)
        return r.text

    def _nearby(self, args: dict) -> str:
        location = args.get("location", "116.397428,39.90923")
        keywords = args.get("keywords", "餐厅")
        radius = args.get("radius", 3000)
        url = "https://restapi.amap.com/v3/place/around"
        params = {"key": self.key, "location": location, "keywords": keywords, "radius": radius}
        r = httpx.get(url, params=params, timeout=10)
        return r.text

    def _geocode(self, args: dict) -> str:
        address = args.get("address", "")
        url = "https://restapi.amap.com/v3/geocode/geo"
        params = {"key": self.key, "address": address}
        r = httpx.get(url, params=params, timeout=10)
        return r.text

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
