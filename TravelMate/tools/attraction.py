import json
from tools.base import BaseTool
from core.rag import search


class AttractionTool(BaseTool):
    name = "attraction"
    description = "景点推荐（基于RAG知识库语义检索）"

    ATTRACTION_DB = {
        "东京": [
            {"name": "浅草寺", "type": "寺庙", "rating": 4.5, "price": "免费", "duration": "1-2小时", "best_time": "8:00-10:00(避人流)", "highlight": "东京最古老寺庙，雷门是地标", "area": "台东区"},
            {"name": "东京晴空塔", "type": "观景台", "rating": 4.3, "price": "2100日元", "duration": "1-2小时", "best_time": "日落时段", "highlight": "634米日本最高建筑，350m+450m双层展望", "area": "墨田区"},
            {"name": "明治神宫", "type": "神社", "rating": 4.6, "price": "免费", "duration": "1小时", "best_time": "清晨(参道最静)", "highlight": "都市中的森林神社，70万棵树", "area": "涩谷区"},
            {"name": "teamLab Borderless", "type": "数字艺术", "rating": 4.7, "price": "3200日元", "duration": "2-3小时", "best_time": "工作日", "highlight": "沉浸式数字艺术博物馆", "area": "台场"},
            {"name": "筑地场外市场", "type": "美食市场", "rating": 4.4, "price": "免费(逛)", "duration": "2-3小时", "best_time": "6:30-10:00", "highlight": "全球最大鱼市场，寿司/海鲜丼/玉子烧", "area": "中央区"},
            {"name": "涩谷十字路口", "type": "城市景观", "rating": 4.2, "price": "免费", "duration": "30分钟", "best_time": "周五晚", "highlight": "世界最繁忙路口，3000人同时通过", "area": "涩谷区"},
            {"name": "秋叶原", "type": "电子/动漫", "rating": 4.3, "price": "免费(逛)", "duration": "3-4小时", "best_time": "工作日", "highlight": "电器+手办+女仆咖啡文化圣地", "area": "千代田区"},
            {"name": "镰仓大佛", "type": "历史", "rating": 4.5, "price": "200日元", "duration": "1小时", "best_time": "上午", "highlight": "13世纪青铜大佛，可进入内部参观", "area": "镰仓市"},
        ],
        "巴黎": [
            {"name": "卢浮宫", "type": "博物馆", "rating": 4.8, "price": "€17", "duration": "3-5小时", "best_time": "周三周五夜场", "highlight": "世界最大博物馆，蒙娜丽莎+胜利女神+断臂维纳斯", "area": "1区"},
            {"name": "埃菲尔铁塔", "type": "地标", "rating": 4.6, "price": "€26(顶楼)", "duration": "2小时", "best_time": "日落前后", "highlight": "巴黎永恒地标，324米", "area": "7区"},
            {"name": "奥赛博物馆", "type": "博物馆", "rating": 4.7, "price": "€16", "duration": "2-3小时", "best_time": "周四夜场", "highlight": "印象派殿堂，莫奈睡莲+梵高自画像", "area": "7区"},
            {"name": "圣心大教堂", "type": "教堂", "rating": 4.5, "price": "免费", "duration": "1小时", "best_time": "日落", "highlight": "蒙马特山上的白色大教堂，巴黎全景", "area": "18区"},
            {"name": "凡尔赛宫", "type": "宫殿", "rating": 4.7, "price": "€21", "duration": "半天", "best_time": "周二至周五", "highlight": "路易十四的皇宫，镜厅+花园", "area": "凡尔赛"},
        ],
        "曼谷": [
            {"name": "大皇宫", "type": "宫殿", "rating": 4.5, "price": "500泰铢", "duration": "2-3小时", "best_time": "8:30开门即入", "highlight": "泰国最神圣场所，玉佛寺+皇家建筑群", "area": "旧城"},
            {"name": "卧佛寺", "type": "寺庙", "rating": 4.4, "price": "200泰铢", "duration": "1-2小时", "best_time": "下午", "highlight": "46米卧佛，泰式按摩发源地", "area": "旧城"},
            {"name": "恰图恰周末市场", "type": "市场", "rating": 4.3, "price": "免费(逛)", "duration": "3-4小时", "best_time": "周六日上午", "highlight": "15000+摊位，全球最大周末市场", "area": "乍图恰"},
        ],
    }

    def run(self, args: dict) -> str:
        query = args.get("query", args.get("city", "东京 景点"))
        city = args.get("city", "")

        # Combine RAG search with structured data
        rag_results = search(query, 5)
        structured = self.ATTRACTION_DB.get(city, [])

        if not structured:
            # Try to find city from query
            for c in self.ATTRACTION_DB:
                if c in query:
                    structured = self.ATTRACTION_DB[c]
                    city = c
                    break

        output = {
            "city": city or "目的地",
            "rag_results": [{"content": r["content"][:200], "city": r["city"], "score": r.get("score", 0)} for r in rag_results[:3]],
            "attractions": structured if structured else self._generic_attractions(city or query),
            "tips": [
                "热门景点建议网上提前购票，可省1-2小时排队",
                "早上8-9点是参观最佳时段，人少光线好",
                "许多博物馆有免费日，提前查询可省门票费",
            ],
        }
        return json.dumps(output, ensure_ascii=False)

    def _generic_attractions(self, city):
        return [
            {"name": f"{city}历史老城区", "type": "历史", "rating": 4.3, "price": "免费", "duration": "2-3小时", "best_time": "上午", "highlight": "体验当地历史文化", "area": "市中心"},
            {"name": f"{city}中央市场", "type": "市场", "rating": 4.2, "price": "免费(逛)", "duration": "1-2小时", "best_time": "上午", "highlight": "当地美食和特色商品", "area": "市中心"},
            {"name": f"{city}标志性观景点", "type": "观景", "rating": 4.4, "price": "视景点", "duration": "1小时", "best_time": "日落", "highlight": "城市全景", "area": "高地/塔楼"},
        ]
