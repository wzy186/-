import json
from tools.base import BaseTool


class BudgetTool(BaseTool):
    name = "budget"
    description = "预算计算与智能分配"

    DESTINATION_RATES = {
        "东京": {"accommodation_mult": 1.2, "food_mult": 1.3, "transport_mult": 1.1, "attraction_mult": 1.0, "currency": "JPY", "cny_rate": 20.8},
        "巴黎": {"accommodation_mult": 1.4, "food_mult": 1.5, "transport_mult": 1.0, "attraction_mult": 1.2, "currency": "EUR", "cny_rate": 7.85},
        "曼谷": {"accommodation_mult": 0.4, "food_mult": 0.3, "transport_mult": 0.3, "attraction_mult": 0.4, "currency": "THB", "cny_rate": 5.0},
        "巴厘岛": {"accommodation_mult": 0.3, "food_mult": 0.25, "transport_mult": 0.3, "attraction_mult": 0.3, "currency": "IDR", "cny_rate": 2174},
        "伦敦": {"accommodation_mult": 1.5, "food_mult": 1.4, "transport_mult": 1.2, "attraction_mult": 1.3, "currency": "GBP", "cny_rate": 9.15},
        "首尔": {"accommodation_mult": 0.9, "food_mult": 0.8, "transport_mult": 0.7, "attraction_mult": 0.8, "currency": "KRW", "cny_rate": 189},
        "纽约": {"accommodation_mult": 1.6, "food_mult": 1.3, "transport_mult": 0.9, "attraction_mult": 1.2, "currency": "USD", "cny_rate": 7.25},
        "悉尼": {"accommodation_mult": 1.3, "food_mult": 1.2, "transport_mult": 0.9, "attraction_mult": 1.1, "currency": "AUD", "cny_rate": 4.70},
        "迪拜": {"accommodation_mult": 1.5, "food_mult": 1.0, "transport_mult": 0.8, "attraction_mult": 1.5, "currency": "AED", "cny_rate": 1.97},
        "罗马": {"accommodation_mult": 1.1, "food_mult": 1.1, "transport_mult": 0.9, "attraction_mult": 1.1, "currency": "EUR", "cny_rate": 7.85},
    }

    STYLE_ADJUST = {
        "穷游": {"flight": 0.7, "accommodation": 0.4, "food": 0.5, "transport": 0.6, "attraction": 0.5, "shopping": 0.3, "emergency": 0.8},
        "舒适": {"flight": 1.0, "accommodation": 1.0, "food": 1.0, "transport": 1.0, "attraction": 1.0, "shopping": 1.0, "emergency": 1.0},
        "豪华": {"flight": 1.5, "accommodation": 2.5, "food": 2.0, "transport": 1.8, "attraction": 1.5, "shopping": 2.0, "emergency": 1.5},
    }

    def run(self, args: dict) -> str:
        budget = float(args.get("budget", 15000))
        days = int(args.get("days", 5))
        destination = args.get("destination", "东京")
        style = args.get("style", "舒适")
        currency = args.get("currency", "CNY")

        rates = self.DESTINATION_RATES.get(destination, {"accommodation_mult": 1.0, "food_mult": 1.0, "transport_mult": 1.0, "attraction_mult": 1.0, "currency": "USD", "cny_rate": 7.0})
        style_adj = self.STYLE_ADJUST.get(style, self.STYLE_ADJUST["舒适"])

        base_pcts = {"flight": 0.28, "accommodation": 0.25, "food": 0.15, "transport": 0.10, "attraction": 0.10, "shopping": 0.07, "emergency": 0.05}
        adjusted = {}
        total_adj = 0
        for cat, pct in base_pcts.items():
            adj = pct * style_adj.get(cat, 1.0) * rates.get(f"{cat}_mult", 1.0)
            adjusted[cat] = adj
            total_adj += adj

        # Normalize
        for cat in adjusted:
            adjusted[cat] /= total_adj

        icons = {"flight": "✈️", "accommodation": "🏨", "food": "🍜", "transport": "🚃", "attraction": "🎫", "shopping": "🛍️", "emergency": "🛡️"}
        names = {"flight": "机票", "accommodation": "住宿", "food": "餐饮", "transport": "当地交通", "attraction": "门票/体验", "shopping": "购物", "emergency": "应急储备"}
        colors = {"flight": "#6366f1", "accommodation": "#8b5cf6", "food": "#ec4899", "transport": "#f97316", "attraction": "#22c55e", "shopping": "#06b6d4", "emergency": "#94a3b8"}
        tips_map = {
            "flight": "提前45-60天订票最便宜，周二周三出发优惠多",
            "accommodation": "Booking.com免费取消预订，入住当天比价再定",
            "food": "当地市场美食便宜又地道，午餐套餐比晚餐便宜30%",
            "transport": "买交通日票/周票最划算，出租建议用Grab/Uber",
            "attraction": "很多景点有免费日，城市Pass可省30%",
            "shopping": "退税政策要了解，当地超市买手信最便宜",
            "emergency": "预留的应急资金不要动，买份旅行保险更安心",
        }

        allocations = []
        for cat in ["flight", "accommodation", "food", "transport", "attraction", "shopping", "emergency"]:
            pct = adjusted[cat]
            amount = round(budget * pct)
            allocations.append({
                "category": names[cat], "category_key": cat, "amount": amount,
                "percent": round(pct * 100, 1), "color": colors[cat], "icon": icons[cat],
                "tip": tips_map[cat],
            })

        daily_budget_cny = round(budget / days)
        local_currency = rates["currency"]
        local_rate = rates["cny_rate"]
        daily_local = round(daily_budget_cny * local_rate)

        return json.dumps({
            "total": round(budget), "currency": currency, "destination": destination,
            "days": days, "style": style,
            "allocations": allocations,
            "daily_budget": daily_budget_cny,
            "daily_local": daily_local,
            "local_currency": local_currency,
            "local_rate": local_rate,
            "tips": [
                "每天记账，超出日预算时压缩餐饮开支",
                "建议出发前在国内银行换汇，汇率优于机场",
                "应急储备不轻易动用，留作突发情况",
                f"当地货币: {local_currency}，1 CNY ≈ {local_rate} {local_currency}",
            ],
        }, ensure_ascii=False)
