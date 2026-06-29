import json
import random
from datetime import datetime, timedelta
from tools.base import BaseTool


class WeatherTool(BaseTool):
    name = "weather"
    description = "查询目的地天气（7日预报+穿衣建议）"

    WEATHER_DB = {
        "东京": {"base_temp": 28, "base_humidity": 65, "base_uv": 7, "condition_pool": ["晴", "晴转多云", "多云", "阵雨"], "rain_prob": 0.3},
        "巴黎": {"base_temp": 22, "base_humidity": 55, "base_uv": 5, "condition_pool": ["多云", "晴", "阴", "小雨"], "rain_prob": 0.35},
        "曼谷": {"base_temp": 35, "base_humidity": 85, "base_uv": 9, "condition_pool": ["雷阵雨", "晴", "多云", "暴雨"], "rain_prob": 0.5},
        "巴厘岛": {"base_temp": 30, "base_humidity": 75, "base_uv": 10, "condition_pool": ["晴", "多云", "阵雨"], "rain_prob": 0.25},
        "伦敦": {"base_temp": 18, "base_humidity": 70, "base_uv": 3, "condition_pool": ["阴", "小雨", "多云", "晴"], "rain_prob": 0.45},
        "首尔": {"base_temp": 26, "base_humidity": 60, "base_uv": 6, "condition_pool": ["晴", "多云", "阵雨", "闷热"], "rain_prob": 0.3},
        "纽约": {"base_temp": 28, "base_humidity": 55, "base_uv": 7, "condition_pool": ["晴", "多云", "雷阵雨"], "rain_prob": 0.25},
        "悉尼": {"base_temp": 16, "base_humidity": 50, "base_uv": 4, "condition_pool": ["晴", "多云", "风大"], "rain_prob": 0.2},
        "迪拜": {"base_temp": 42, "base_humidity": 30, "base_uv": 11, "condition_pool": ["晴", "酷热", "沙尘"], "rain_prob": 0.02},
        "罗马": {"base_temp": 30, "base_humidity": 45, "base_uv": 8, "condition_pool": ["晴", "晴热", "多云"], "rain_prob": 0.1},
        "大阪": {"base_temp": 29, "base_humidity": 68, "base_uv": 7, "condition_pool": ["晴", "多云", "闷热", "阵雨"], "rain_prob": 0.3},
        "京都": {"base_temp": 30, "base_humidity": 65, "base_uv": 7, "condition_pool": ["晴", "多云", "闷热"], "rain_prob": 0.25},
    }

    CLOTHING_ADVICE = {
        (0, 10): {"clothes": "厚外套/羽绒服+毛衣+保暖内衣", "extras": "围巾·手套·帽子·暖宝宝", "umbrella": "折叠伞必备"},
        (10, 20): {"clothes": "薄外套/风衣+长袖衬衫", "extras": "薄围巾备用", "umbrella": "折叠伞建议带"},
        (20, 28): {"clothes": "T恤+轻薄外套（早晚用）", "extras": "防晒霜SPF30+", "umbrella": "折叠伞建议带"},
        (28, 35): {"clothes": "短袖+轻薄透气衣物", "extras": "防晒霜SPF50+·遮阳帽·墨镜", "umbrella": "折叠伞+雨衣"},
        (35, 50): {"clothes": "最轻薄的透气衣物", "extras": "防晒霜SPF50+·遮阳帽·墨镜·补水", "umbrella": "防晒伞+折叠伞"},
    }

    def run(self, args: dict) -> str:
        city = args.get("city", args.get("destination", "东京"))
        days = min(int(args.get("days", 7)), 7)
        db = self.WEATHER_DB.get(city, {"base_temp": 22 + random.randint(0, 10), "base_humidity": 60, "base_uv": 5, "condition_pool": ["多云", "晴"], "rain_prob": 0.3})

        forecast = []
        for d in range(days):
            date = (datetime.now() + timedelta(days=d)).strftime("%m/%d")
            temp_high = db["base_temp"] + random.randint(-3, 3)
            temp_low = temp_high - random.randint(5, 10)
            condition = random.choice(db["condition_pool"])
            rain = random.randint(10, 90) if random.random() < db["rain_prob"] else random.randint(0, 20)
            humidity = db["base_humidity"] + random.randint(-10, 10)
            uv = max(1, db["base_uv"] + random.randint(-2, 2))

            hourly = []
            for h in [8, 12, 15, 18, 21]:
                h_temp = temp_low + (temp_high - temp_low) * (1 - abs(h - 14) / 10)
                hourly.append({"time": f"{h:02d}:00", "temp": f"{int(h_temp)}°C", "condition": condition if h < 19 else random.choice(["晴", "多云", condition])})

            forecast.append({
                "date": date, "temp_high": f"{temp_high}°C", "temp_low": f"{temp_low}°C",
                "condition": condition, "rain_probability": f"{rain}%", "humidity": f"{humidity}%",
                "uv_index": uv, "uv_label": self._uv_label(uv), "hourly": hourly,
            })

        avg_temp = db["base_temp"]
        for low, high in self.CLOTHING_ADVICE:
            if low <= avg_temp < high:
                clothing = self.CLOTHING_ADVICE[(low, high)]
                break
        else:
            clothing = self.CLOTHING_ADVICE[(28, 35)]

        today = forecast[0]
        result = {
            "city": city, "forecast": forecast,
            "today": today,
            "clothing": clothing,
            "suggestion": self._suggestion(city, today),
            "health_tips": self._health_tips(today, city),
        }
        return json.dumps(result, ensure_ascii=False)

    def _uv_label(self, uv):
        if uv <= 2: return "弱"
        if uv <= 5: return "中等"
        if uv <= 7: return "强"
        if uv <= 10: return "很强"
        return "极强"

    def _suggestion(self, city, today):
        suggestions = {
            "东京": "建议穿轻便透气衣物，随身带折叠伞。便利店有卖透明伞500日元。",
            "巴黎": "早晚温差大，建议带薄外套。下雨频率高，伞是标配。",
            "曼谷": "防晒防蚊必备，随时可能暴雨。室内空调很冷，带薄外套。",
            "巴厘岛": "防晒霜SPF50+，泳衣必带。傍晚可能有阵雨但很快放晴。",
            "伦敦": "必带雨伞和薄外套，一天经历四季不是传说。",
            "首尔": "夏季闷热多雨，冬季极冷需羽绒服。春秋最宜出行。",
            "纽约": "夏季湿热，冬季严寒。地铁没空调时很热，室内冷气又很足。",
        }
        return suggestions.get(city, "建议查看出发前3天天气预报，做好灵活准备。")

    def _health_tips(self, today, city):
        tips = []
        uv = int(today.get("uv_index", 5))
        if uv >= 7:
            tips.append("紫外线强，户外活动每2小时补涂防晒霜")
        if int(today.get("rain_probability", "0%").replace("%", "")) > 50:
            tips.append("降雨概率高，建议室内活动备选方案")
        temp = int(today.get("temp_high", "25").replace("°C", ""))
        if temp >= 35:
            tips.append("高温预警，注意补水防中暑，避免正午户外活动")
        if temp <= 5:
            tips.append("低温预警，注意保暖防感冒，室内外温差大")
        humidity = int(today.get("humidity", "60%").replace("%", ""))
        if humidity >= 80:
            tips.append("湿度高，衣物不易干，建议带速干面料")
        if not tips:
            tips.append("天气适宜出行，享受旅途！")
        return tips
