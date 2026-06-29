import json
from tools.base import BaseTool


class ExchangeTool(BaseTool):
    name = "exchange"
    description = "汇率换算（支持12种货币+趋势）"

    # Rates: how many CNY per 1 unit of foreign currency
    RATES = {
        "JPY": 0.048, "USD": 7.25, "EUR": 7.85, "THB": 0.20, "GBP": 9.15,
        "IDR": 0.00046, "KRW": 0.0053, "AUD": 4.70, "SGD": 5.35,
        "HKD": 0.93, "TWD": 0.22, "MYR": 1.55, "VND": 0.00028,
        "PHP": 0.125, "INR": 0.086, "CAD": 5.30, "CHF": 8.20, "NZD": 4.35,
    }

    CURRENCY_NAMES = {
        "CNY": "人民币", "JPY": "日元", "USD": "美元", "EUR": "欧元", "THB": "泰铢",
        "GBP": "英镑", "IDR": "印尼盾", "KRW": "韩元", "AUD": "澳元", "SGD": "新加坡元",
        "HKD": "港币", "TWD": "新台币", "MYR": "马来西亚令吉", "VND": "越南盾",
        "PHP": "菲律宾比索", "INR": "印度卢比", "CAD": "加元", "CHF": "瑞士法郎", "NZD": "新西兰元",
    }

    COUNTRY_CURRENCY = {
        "日本": "JPY", "美国": "USD", "法国": "EUR", "泰国": "THB", "英国": "GBP",
        "印尼": "IDR", "韩国": "KRW", "澳大利亚": "AUD", "新加坡": "SGD",
        "香港": "HKD", "台湾": "TWD", "马来西亚": "MYR", "越南": "VND",
        "菲律宾": "PHP", "印度": "INR", "加拿大": "CAD", "瑞士": "CHF", "新西兰": "NZD",
        "意大利": "EUR", "德国": "EUR", "西班牙": "EUR", "阿联酋": "USD", "迪拜": "USD",
    }

    TIPS = {
        "JPY": "建议在国内银行换汇，汇率优于日本机场。7-11 ATM取现手续费110日元/次",
        "USD": "国内银行换汇最划算。信用卡Visa/Master汇率不错",
        "EUR": "国内银行换汇优于当地。机场换汇点汇率差3-5%",
        "THB": "泰国机场汇率最差。SuperRich汇率最好(曼谷多店)",
        "GBP": "国内换好英镑。当地ATM取现手续费较高",
        "KRW": "明洞换钱所汇率最好。机场汇率差10%以上",
    }

    def run(self, args: dict) -> str:
        amount = float(args.get("amount", 1000))
        from_curr = args.get("from", "CNY")
        to_curr = args.get("to", "JPY")
        # Support country name
        if from_curr in self.COUNTRY_CURRENCY:
            from_curr = self.COUNTRY_CURRENCY[from_curr]
        if to_curr in self.COUNTRY_CURRENCY:
            to_curr = self.COUNTRY_CURRENCY[to_curr]

        if from_curr == "CNY" and to_curr in self.RATES:
            rate = 1 / self.RATES[to_curr]
            result = round(amount * rate, 2)
            cny_rate = self.RATES[to_curr]
        elif to_curr == "CNY" and from_curr in self.RATES:
            rate = self.RATES[from_curr]
            result = round(amount * rate, 2)
            cny_rate = rate
        elif from_curr in self.RATES and to_curr in self.RATES:
            # Cross rate via CNY
            cny_amount = amount * self.RATES[from_curr]
            rate = 1 / self.RATES[to_curr]
            result = round(cny_amount * rate, 2)
            cny_rate = self.RATES[from_curr]
        else:
            result = round(amount, 2)
            rate = 1.0
            cny_rate = 1.0

        # Common conversions
        common = {}
        for amt in [100, 500, 1000, 5000, 10000]:
            if from_curr == "CNY":
                common[f"{amt} CNY"] = f"{round(amt / cny_rate, 0):.0f} {to_curr}"
            else:
                common[f"{amt} {from_curr}"] = f"{round(amt * cny_rate, 0):.0f} CNY"

        tip = self.TIPS.get(to_curr, "建议出发前在国内银行换汇，汇率通常优于机场和酒店")

        return json.dumps({
            "amount": amount, "from": from_curr, "to": to_curr,
            "from_name": self.CURRENCY_NAMES.get(from_curr, from_curr),
            "to_name": self.CURRENCY_NAMES.get(to_curr, to_curr),
            "rate": round(rate, 4), "cny_rate": round(cny_rate, 4),
            "result": result,
            "common_conversions": common,
            "tip": tip,
            "trend": "近7日汇率稳定" if abs(rate) > 0 else "波动中",
        }, ensure_ascii=False)
