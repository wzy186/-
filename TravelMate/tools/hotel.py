import json
import random
from tools.base import BaseTool


class HotelTool(BaseTool):
    name = "hotel"
    description = "酒店推荐（按星级/价格/区域筛选）"

    HOTELS = {
        "东京": [
            {"name": "新宿格兰贝尔酒店", "area": "新宿", "price": 600, "rating": 4.3, "stars": 4, "type": "商务", "amenities": ["WiFi", "自助洗衣", "大浴场"], "nearby": "新宿站步行5分", "breakfast": "含(自助)", "review_highlight": "位置极佳，新宿站直连"},
            {"name": "浅草豪景酒店", "area": "浅草", "price": 450, "rating": 4.1, "stars": 4, "type": "商务", "amenities": ["WiFi", "展望台"], "nearby": "浅草站步行3分", "breakfast": "含(和式)", "review_highlight": "浅草寺步行可达，性价比高"},
            {"name": "Khaosan Tokyo", "area": "浅草", "price": 150, "rating": 4.0, "stars": 1, "type": "青旅", "amenities": ["WiFi", "厨房", "公共休息室"], "nearby": "浅草站步行5分", "breakfast": "无", "review_highlight": "背包客天堂，社交氛围好"},
            {"name": "安缦东京", "area": "大手町", "price": 5000, "rating": 4.9, "stars": 5, "type": "奢华", "amenities": ["WiFi", "SPA", "泳池", "私人管家"], "nearby": "大手町站直连", "breakfast": "含(豪华自助)", "review_highlight": "东京最佳酒店，皇居景观"},
            {"name": "TRUNK(HOTEL)", "area": "涩谷", "price": 2500, "rating": 4.6, "stars": 5, "type": "设计酒店", "amenities": ["WiFi", "酒吧", "教堂"], "nearby": "涩谷站步行7分", "breakfast": "含", "review_highlight": "潮流设计酒店，适合打卡"},
            {"name": "东京站八重洲大和Roynet", "area": "东京站", "price": 550, "rating": 4.4, "stars": 4, "type": "商务", "amenities": ["WiFi", "大浴场", "自助洗衣"], "nearby": "东京站步行3分", "breakfast": "含", "review_highlight": "东京站旁，交通最便利"},
            {"name": "相铁Fresa Inn 新桥站前", "area": "新桥", "price": 350, "rating": 4.0, "stars": 3, "type": "商务", "amenities": ["WiFi"], "nearby": "新桥站步行1分", "breakfast": "无", "review_highlight": "便宜干净，新桥站1分"},
        ],
        "巴黎": [
            {"name": "Hotel Le Marais", "area": "玛黑区", "price": 800, "rating": 4.2, "stars": 4, "type": "精品", "amenities": ["WiFi", "酒吧"], "nearby": "Saint-Paul站步行3分", "breakfast": "含", "review_highlight": "玛黑区核心位置，文艺氛围"},
            {"name": "Generator Paris", "area": "巴士底", "price": 250, "rating": 3.9, "stars": 1, "type": "青旅", "amenities": ["WiFi", "酒吧", "天台"], "nearby": "Oberkampf站步行5分", "breakfast": "含(简式)", "review_highlight": "设计感青旅，天台看巴黎"},
            {"name": "丽兹巴黎", "area": "旺多姆广场", "price": 6000, "rating": 4.9, "stars": 5, "type": "奢华", "amenities": ["WiFi", "SPA", "米其林餐厅", "Chanel Spa"], "nearby": "Tuileries站步行3分", "breakfast": "含(极致)", "review_highlight": "全球最传奇酒店之一"},
            {"name": "Citadines Saint-Germain", "area": "圣日耳曼", "price": 650, "rating": 4.1, "stars": 4, "type": "公寓式", "amenities": ["WiFi", "厨房", "洗衣"], "nearby": "Saint-Michel站步行5分", "breakfast": "无", "review_highlight": "公寓式酒店，长住首选"},
        ],
        "曼谷": [
            {"name": "Lub d Bangkok Silom", "area": "是隆", "price": 120, "rating": 4.2, "stars": 1, "type": "青旅", "amenities": ["WiFi", "泳池", "酒吧"], "nearby": "BTS Chong Nonsri站步行5分", "breakfast": "含(简式)", "review_highlight": "曼谷最佳青旅，泳池+社交"},
            {"name": "暹罗@Siam设计酒店", "area": "Siam", "price": 400, "rating": 4.5, "stars": 4, "type": "设计酒店", "amenities": ["WiFi", "泳池", "SPA"], "nearby": "BTS National Stadium站步行3分", "breakfast": "含", "review_highlight": "设计感十足，Siam商圈核心"},
            {"name": "曼谷半岛酒店", "area": "湄南河畔", "price": 2500, "rating": 4.8, "stars": 5, "type": "奢华", "amenities": ["WiFi", "泳池", "SPA", "私人游船"], "nearby": "BTS Saphan Taksin站+摆渡船", "breakfast": "含(河景自助)", "review_highlight": "河景无敌，全球最佳半岛"},
        ],
        "伦敦": [
            {"name": "Generator London", "area": "King's Cross", "price": 200, "rating": 3.9, "stars": 1, "type": "青旅", "amenities": ["WiFi", "酒吧"], "nearby": "King's Cross站步行5分", "breakfast": "含(简式)", "review_highlight": "伦敦最佳青旅之一"},
            {"name": "Premier Inn County Hall", "area": "南岸", "price": 900, "rating": 4.2, "stars": 3, "type": "商务", "amenities": ["WiFi"], "nearby": "Waterloo站步行5分", "breakfast": "含", "review_highlight": "伦敦眼旁，性价比高"},
            {"name": "The Savoy", "area": "Strand", "price": 5000, "rating": 4.9, "stars": 5, "type": "奢华", "amenities": ["WiFi", "SPA", "河景下午茶"], "nearby": "Covent Garden步行5分", "breakfast": "含", "review_highlight": "伦敦传奇酒店，Savoy下午茶"},
        ],
    }

    def run(self, args: dict) -> str:
        city = args.get("city", args.get("destination", "东京"))
        budget_per_night = int(args.get("budget_per_night", 800))
        style = args.get("style", "舒适")
        area_pref = args.get("area", "")

        hotels = self.HOTELS.get(city, self._generate_generic(city))

        # Filter by budget (with 20% flexibility)
        max_price = budget_per_night * 1.2
        filtered = [h for h in hotels if h["price"] <= max_price]

        # Filter by area if specified
        if area_pref:
            area_filtered = [h for h in filtered if area_pref in h["area"]]
            if area_filtered:
                filtered = area_filtered

        # Filter by style
        style_map = {"穷游": ["青旅"], "舒适": ["商务", "精品", "公寓式"], "豪华": ["奢华", "设计酒店"]}
        preferred_types = style_map.get(style, [])
        if preferred_types:
            style_filtered = [h for h in filtered if h["type"] in preferred_types]
            if style_filtered:
                filtered = style_filtered

        if not filtered:
            filtered = sorted(hotels, key=lambda x: x["price"])[:3]

        # Sort by rating
        filtered.sort(key=lambda x: x["rating"], reverse=True)

        booking_tips = [
            "Booking.com免费取消预订最灵活",
            "Agoda在亚洲酒店价格更有优势",
            "直接联系酒店官网可能有最佳价格保证",
            "含早餐的酒店通常比不含+外出吃早餐更划算",
        ]

        return json.dumps({
            "city": city, "budget_per_night": budget_per_night, "style": style,
            "hotels": filtered, "count": len(filtered),
            "booking_tips": booking_tips,
            "area_guide": self._area_guide(city),
        }, ensure_ascii=False)

    def _area_guide(self, city):
        guides = {
            "东京": {"推荐区域": ["新宿(交通枢纽)", "浅草(传统氛围)", "涩谷(潮流文化)", "东京站(商务便利)"], "避坑": ["歌舞伎町深夜噪音大", "池袋东口治安稍差"]},
            "巴黎": {"推荐区域": ["玛黑区(文艺)", "圣日耳曼(左岸)", "蒙马特(浪漫)", "拉丁区(平价)"], "避坑": ["18区北部夜间避免", "Gare du Nord周边注意安全"]},
            "曼谷": {"推荐区域": ["Siam(购物)", "是隆(商务)", "考山路(背包客)", "湄南河畔(度假)"], "避坑": ["考山路深夜噪音", "部分巷尾青旅卫生差"]},
        }
        return guides.get(city, {"推荐区域": ["市中心(交通便利)"], "避坑": ["提前查看评价"]})

    def _generate_generic(self, city):
        return [
            {"name": f"{city}中央酒店", "area": "市中心", "price": random.randint(400, 1200), "rating": 4.0, "stars": 4, "type": "商务", "amenities": ["WiFi", "早餐"], "nearby": "中心站步行5分", "breakfast": "含", "review_highlight": "位置好，干净"},
            {"name": f"{city}青年旅舍", "area": "市中心", "price": random.randint(100, 300), "rating": 3.8, "stars": 1, "type": "青旅", "amenities": ["WiFi", "厨房"], "nearby": "中心站步行10分", "breakfast": "无", "review_highlight": "便宜实惠"},
            {"name": f"{city}豪华酒店", "area": "核心区", "price": random.randint(3000, 8000), "rating": 4.8, "stars": 5, "type": "奢华", "amenities": ["WiFi", "SPA", "泳池"], "nearby": "核心区步行3分", "breakfast": "含", "review_highlight": "顶级体验"},
        ]
