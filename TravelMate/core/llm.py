import os
import json
import random
import time
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
AMAP_KEY = os.getenv("AMAP_API_KEY", "")

_client = None

# ── Destination-specific mock data ──

_DESTINATION_PLANS = {
    "东京": {
        "comfortable": {"title": "东京5日文化深度游", "overview": "从浅草寺到秋叶原，从筑地到镰仓，沉浸式体验东京的传统与潮流",
            "days": [
                {"day": 1, "theme": "传统东京", "spots": ["浅草寺（雷门·本堂）", "仲见世商店街（人形烧·煎饼）", "东京晴空塔（350m展望台）", "上野公园（国立博物馆·不忍池）"], "transport": "地铁银座线+步行", "cost": 800, "meals": {"breakfast": "酒店自助", "lunch": "浅草鰻魚飯 ¥2000", "dinner": "上野居酒屋 ¥3000"}, "notes": "浅草寺建议早8点到避开人流，晴空塔日落时段最美"},
                {"day": 2, "theme": "潮流文化", "spots": ["明治神宫（森林步道）", "原宿竹下通（可丽饼·古着）", "表参道（建筑巡礼·Omotesando Hills）", "涩谷十字路口（SCRAMBLE SQUARE顶楼观景）"], "transport": "JR山手线+地铁", "cost": 600, "meals": {"breakfast": "便利店饭团", "lunch": "原宿一蘭拉面 ¥1500", "dinner": "涩谷烧肉 ¥4000"}, "notes": "涩谷SCRAMBLE SQUARE顶楼Roof Shibuya天空观景台免费"},
                {"day": 3, "theme": "美食之旅", "spots": ["筑地场外市场（玉子烧·海胆饭）", "银座（三越百货·歌舞伎座）", "东京塔（大眺望台）", "六本木Hills（森美术馆·城市夜景）"], "transport": "地铁日比谷线+步行", "cost": 1200, "meals": {"breakfast": "筑地寿司大 ¥4000", "lunch": "银座天ぷら ¥2500", "dinner": "六本木铁板烧 ¥6000"}, "notes": "筑地建议6:30到，金枪鱼拍卖旁观需排队。银座周日中央通步行者天国"},
                {"day": 4, "theme": "镰仓一日游", "spots": ["镰仓高校前站（灌篮高手路口）", "长谷寺（镰仓大佛·观景台）", "小町通（鸽子饼干·抹茶冰淇淋）", "江之岛（灯塔·海鲜）"], "transport": "JR湘南新宿线+江之电", "cost": 1000, "meals": {"breakfast": "新宿咖啡厅", "lunch": "镰仓しらす丼 ¥1800", "dinner": "江之岛生しらす料理 ¥3500"}, "notes": "江之电1日券700日元超值。镰仓高校前站拍照注意安全，火车频繁"},
                {"day": 5, "theme": "科技与购物", "spots": ["秋叶原（电器·手办·女仆咖啡）", "台场（高达立像·teamLab）", "东京站（丸之内·GranRoof）", "新宿（回忆横丁·歌舞伎町）"], "transport": "地铁+百合海鸥号", "cost": 1500, "meals": {"breakfast": "秋叶原咖啡厅", "lunch": "台场拉面国技馆 ¥1200", "dinner": "新宿回忆横丁 ¥3000"}, "notes": "秋叶原BicCamera免税+优惠券。台场teamLab需提前预约。新宿回忆横丁17:00后最热闹"},
            ], "total_estimate": 5100, "currency": "CNY", "tips": ["买Suica卡交通最方便，充值押金500日元可退", "便利店Seven ATM取现手续费最低110日元", "餐厅大多不用给小费，居酒屋可能收坐席费", "商场购物满5000日元可办退税，护照随身带"]},
        "luxury": {"title": "东京5日豪华尊享游", "overview": "安缦住宿·米其林盛宴·私人茶道·箱根温泉，体验极致和风奢华",
            "days": [
                {"day": 1, "theme": "奢华入住", "spots": ["安缦东京（大手町·皇居景观套房）", "皇居东御苑（私人导览）", "银座米其林二星L'Effervescence"], "transport": "机场专车接机", "cost": 5000, "meals": {"breakfast": "机上午餐", "lunch": "安缦下午茶", "dinner": "L'Effervescence ¥45000"}, "notes": "安缦东京可远眺皇居，建议选皇居景观房"},
                {"day": 2, "theme": "私享文化", "spots": ["私人茶道体验（表千家流）", "根津美术馆（庭院私人参观）", "表参道Dior旗舰店"], "transport": "专车", "cost": 4000, "meals": {"breakfast": "安缦和式早餐", "lunch": "南青山カフェ ¥5000", "dinner": "虎白（米其林三星）¥38000"}, "notes": "茶道需提前2周预约，虎白至少提前1月预约"},
                {"day": 3, "theme": "米其林之旅", "spots": ["筑地寿司大（VIP席位）", "数寄屋桥次郎（预约制）", "六本木Keyakizaka（铁板烧）"], "transport": "专车", "cost": 6000, "meals": {"breakfast": "安缦早餐", "lunch": "寿司大OMAKASE ¥20000", "dinner": "数寄屋桥次郎 ¥40000"}, "notes": "米其林三星餐厅至少提前1月预约，需信用卡担保"},
                {"day": 4, "theme": "箱根温泉", "spots": ["箱根强罗花扇（露天风吕套房）", "芦之湖（海盗船私人游览）", "大涌谷（黑鸡蛋）"], "transport": "新干线+专车", "cost": 4500, "meals": {"breakfast": "安缦便当", "lunch": "箱根怀石料理 ¥15000", "dinner": "旅馆会席料理含宿"}, "notes": "建议住一晚温泉旅馆，体验正宗露天风吕。强罗花扇有私人温泉"},
                {"day": 5, "theme": "购物收官", "spots": ["银座三越（VIP室·个人购物顾问）", "表参道Hills", "新宿伊势丹"], "transport": "专车", "cost": 3000, "meals": {"breakfast": "旅馆和式早餐", "lunch": "银座割烹 ¥12000", "dinner": "新叙叙苑烧肉 ¥20000"}, "notes": "百货公司退税柜台统一办理，满50万日元有额外折扣"},
            ], "total_estimate": 22500, "currency": "CNY", "tips": ["高级餐厅需信用卡担保预约", "温泉旅馆有着装要求，带浴衣即可", "购物满5000日元可退税，护照必须随身带", "专车建议提前1天预约，Japan Taxi App可叫"]},
        "budget": {"title": "东京5日穷游攻略", "overview": "免费景点+便利店美食+青旅住宿，用最少预算体验最地道的东京",
            "days": [
                {"day": 1, "theme": "免费景点日", "spots": ["明治神宫（免费·森林参道）", "代代木公园（免费·野餐）", "涩谷十字路口（免费·打卡）", "新宿御苑（500日元·日式庭园）"], "transport": "地铁一日券600日元", "cost": 200, "meals": {"breakfast": "便利店饭团¥110", "lunch": "松屋牛丼¥400", "dinner": "便利店便当¥500"}, "notes": "地铁一日券600日元，坐3次就回本。新宿御苑周一体休"},
                {"day": 2, "theme": "平价文化日", "spots": ["浅草寺（免费·雷门）", "上野公园（免费·不忍池）", "国立博物馆（大学生免费/一般1000日元）", "秋叶原（免费逛·百元店）"], "transport": "地铁", "cost": 300, "meals": {"breakfast": "面包店¥200", "lunch": "上野阿美横町定食¥800", "dinner": "便利店沙拉+饭团¥500"}, "notes": "便利店饭团100日元一个。百元店大创可买旅行用品"},
                {"day": 3, "theme": "筑地+台场", "spots": ["筑地场外市场（免费逛·试吃）", "台场海滨公园（免费·自由女神像）", "Digital Art Museum（3200日元·teamLab）"], "transport": "地铁+百合海鸥号", "cost": 400, "meals": {"breakfast": "筑场玉子烧¥200", "lunch": "筑地海鲜丼¥1500", "dinner": "台场拉面¥900"}, "notes": "筑地早市有低价试吃。台场teamLab周一休馆"},
                {"day": 4, "theme": "镰仓省钱版", "spots": ["镰仓大佛（200日元·可入内部）", "江之岛（免费·灯塔下层）", "小町通（免费逛·鸽子饼干试吃）", "镰仓高校前站（免费·拍照圣地）"], "transport": "JR镰仓·江之电周游券700日元", "cost": 500, "meals": {"breakfast": "便利店", "lunch": "镰仓しらす丼¥1200", "dinner": "便利店¥500"}, "notes": "周游券含江之电无限乘坐。镰仓高校前拍照注意安全"},
                {"day": 5, "theme": "秋叶原+药妆", "spots": ["秋叶原（免费逛·女仆咖啡体验）", "唐吉诃德（深夜打折）", "池袋Sunshine City（免费·水族馆另票）"], "transport": "地铁", "cost": 400, "meals": {"breakfast": "便利店", "lunch": "秋叶原咖喱¥700", "dinner": "唐吉诃德便当¥400"}, "notes": "唐吉诃德深夜打折力度大。药妆店比价用kakaku.com"},
            ], "total_estimate": 1800, "currency": "CNY", "tips": ["住青旅每晚2000-3000日元，Khaosan Tokyo推荐", "7-11的ATM取现手续费最低110日元", "百元店大创可买旅行用品，质量不错", "便利店美食超出预期：饭团、三明治、甜品都好吃"]},
    },
    "巴黎": {
        "comfortable": {"title": "巴黎5日艺术浪漫之旅", "overview": "从卢浮宫到塞纳河，从蒙马特到凡尔赛，感受光之城的浪漫与艺术",
            "days": [
                {"day": 1, "theme": "经典巴黎", "spots": ["埃菲尔铁塔（二层展望台）", "塞纳河游船（Bateaux Mouches）", "战神广场（铁塔最佳角度）", "荣军院（拿破仑墓）"], "transport": "Metro+步行", "cost": 1200, "meals": {"breakfast": "酒店早餐", "lunch": "左岸咖啡馆 ¥200", "dinner": "塞纳河畔法餐 ¥600"}, "notes": "埃菲尔铁塔建议提前网上购票。塞纳河游船日落时段最美"},
                {"day": 2, "theme": "艺术殿堂", "spots": ["卢浮宫（蒙娜丽莎·胜利女神）", "杜乐丽花园", "奥赛博物馆（印象派殿堂）", "圣日耳曼德佩区"], "transport": "Metro", "cost": 800, "meals": {"breakfast": "面包房羊角面包", "lunch": "博物馆咖啡厅 ¥150", "dinner": "拉丁区法式小馆 ¥300"}, "notes": "卢浮宫每月第一个周日免费。奥赛五楼时钟咖啡厅必去"},
                {"day": 3, "theme": "蒙马特与购物", "spots": ["圣心大教堂", "蒙马特画家广场", "老佛爷百货", "巴黎歌剧院"], "transport": "Metro+步行", "cost": 1000, "meals": {"breakfast": "蒙马特咖啡馆", "lunch": "小丘广场可丽饼 ¥80", "dinner": "蒙马特法餐 ¥400"}, "notes": "老佛爷百货12%退税。蒙马特注意随身物品防扒手"},
                {"day": 4, "theme": "凡尔赛一日游", "spots": ["凡尔赛宫（镜厅·花园）", "大小特里亚农宫", "玛丽·安托瓦内特庄园"], "transport": "RER C线", "cost": 600, "meals": {"breakfast": "巴黎面包房", "lunch": "凡尔赛宫内餐厅 ¥200", "dinner": "回巴黎后铁塔附近 ¥350"}, "notes": "凡尔赛宫每月第一个周日免费。花园比宫殿更值得细看"},
                {"day": 5, "theme": "玛黑与告别", "spots": ["蓬皮杜中心", "玛黑区（画廊·古着）", "孚日广场", "巴黎圣母院（外观·修复中）"], "transport": "Metro", "cost": 500, "meals": {"breakfast": "酒店早餐", "lunch": "玛黑区犹太餐厅 ¥180", "dinner": "左岸米其林推荐 ¥500"}, "notes": "蓬皮杜每月第一个周日免费。玛黑区周日很多店开"},
            ], "total_estimate": 4100, "currency": "CNY", "tips": ["买Paris Museum Pass 2日€55，省排队时间", "地铁买carnet 10张票€16.9比单买便宜", "餐厅小费5-10%即可，已含服务费", "注意防扒手，地铁和旅游景点是高发区"]},
        "luxury": {"title": "巴黎5日奢华尊享游", "overview": "丽兹住宿·米其林三星·私人导览·塞纳河私人游船，法式极致浪漫",
            "days": [
                {"day": 1, "theme": "奢华抵达", "spots": ["丽兹巴黎（旺多姆广场套房）", "旺多姆广场（高级珠宝）", "Le Cinq（米其林三星晚餐）"], "transport": "戴高乐专车接机", "cost": 8000, "meals": {"breakfast": "机上", "lunch": "丽兹L'Espadon", "dinner": "Le Cinq ¥8000"}, "notes": "丽兹巴黎Coco Chanel套房最经典"},
                {"day": 2, "theme": "私人艺术", "spots": ["卢浮宫私人导览（2小时·闭馆后）", "私人塞纳河游船", "蒙马特私人画室体验"], "transport": "专车", "cost": 6000, "meals": {"breakfast": "丽兹早餐", "lunch": "左岸米其林推荐 ¥3000", "dinner": "Pierre Gagnaire（三星）¥6000"}, "notes": "私人导览需提前2周预约"},
                {"day": 3, "theme": "凡尔赛私人", "spots": ["凡尔赛宫私人导览（含闭馆区域）", "橘园私人音乐会", "香奈儿旗舰店VIP"], "transport": "专车", "cost": 5000, "meals": {"breakfast": "丽兹早餐", "lunch": "凡尔赛Ore餐厅 ¥2000", "dinner": "Epicure（三星）¥7000"}, "notes": "凡尔赛私人导览可进入不对公众开放的王后卧室"},
                {"day": 4, "theme": "香槟之旅", "spots": ["兰斯大教堂", "Dom Pérignon酒窖私人品鉴", "香槟区庄园午餐"], "transport": "专车（2小时）", "cost": 4000, "meals": {"breakfast": "丽兹早餐", "lunch": "香槟区庄园 ¥2500", "dinner": "回巴黎Alain Ducasse ¥5000"}, "notes": "Dom Pérignon品鉴需提前预约"},
                {"day": 5, "theme": "购物收官", "spots": ["爱马仕总店", "蒙田大道（Dior·Chanel）", "Ladurée（马卡龙礼盒）"], "transport": "专车", "cost": 3000, "meals": {"breakfast": "丽兹早餐", "lunch": "蒙田大道Le Relais ¥2000", "dinner": "丽兹Bar Hemingway告别酒"}, "notes": "非EU居民退税12-14%"},
            ], "total_estimate": 26000, "currency": "CNY", "tips": ["米其林三星至少提前2月预约", "丽兹Bar Hemingway是海明威最爱", "法国高级餐厅有着装要求", "非EU居民退税最高14%"]},
        "budget": {"title": "巴黎5日穷游攻略", "overview": "免费博物馆日·面包房美食·青旅住宿，用最少预算感受浪漫之都",
            "days": [
                {"day": 1, "theme": "免费巴黎", "spots": ["埃菲尔铁塔（地面层免费·拍照）", "塞纳河岸边（免费·野餐）", "圣心大教堂（免费）", "蒙马特画家广场（免费）"], "transport": "Metro单程票€2.15", "cost": 300, "meals": {"breakfast": "面包房羊角¥15", "lunch": "法棍三明治¥25", "dinner": "超市熟食¥30"}, "notes": "铁塔下战神广场野餐最省钱。圣心教堂免费，登顶€7"},
                {"day": 2, "theme": "免费博物馆日", "spots": ["卢浮宫（每月第一个周日免费）", "蓬皮杜中心（每月第一个周日免费）", "杜乐丽花园（免费）"], "transport": "Metro", "cost": 200, "meals": {"breakfast": "面包房¥15", "lunch": "拉丁区kebab¥50", "dinner": "超市¥25"}, "notes": "每月第一个周日多个博物馆免费。26岁以下EU学生很多馆免费"},
                {"day": 3, "theme": "步行巴黎", "spots": ["玛黑区（免费逛·古着店）", "孚日广场（免费·维克多雨果故居免费）", "塞纳河旧书摊（免费逛）", "巴黎圣母院（外观·修复中）"], "transport": "步行", "cost": 150, "meals": {"breakfast": "便利店¥12", "lunch": "玛黑区falafel¥80", "dinner": "超市¥25"}, "notes": "玛黑区L'As du Fallafel排长队但值得"},
                {"day": 4, "theme": "凡尔赛省钱", "spots": ["凡尔赛宫（每月第一个周日免费）", "凡尔赛花园（免费·比宫殿还美）", "凡尔赛小镇（免费逛）"], "transport": "RER C线€4.2", "cost": 250, "meals": {"breakfast": "面包房¥15", "lunch": "凡尔赛面包¥20", "dinner": "回巴黎超市¥25"}, "notes": "RER C线到凡尔赛最便宜。花园免费时段比宫殿更值得"},
                {"day": 5, "theme": "免费告别", "spots": ["卢森堡公园（免费）", "圣日耳曼德佩区（免费逛）", "花市（免费逛·周日鸟市）"], "transport": "Metro", "cost": 100, "meals": {"breakfast": "面包房¥15", "lunch": "可丽饼摊¥50", "dinner": "超市¥25"}, "notes": "卢森堡公园巴黎人最爱的休闲地"},
            ], "total_estimate": 1000, "currency": "CNY", "tips": ["住青旅/St Christopher's每晚€25-35", "超市家乐福City最便宜", "自来水可直接饮用，带水瓶省钱", "Metro 10次票carnet€16.9比单买便宜"]},
    },
    "曼谷": {
        "comfortable": {"title": "曼谷5日热带风情游", "overview": "大皇宫到考山路，水上市场到夜市美食，体验微笑之国的热情",
            "days": [
                {"day": 1, "theme": "经典曼谷", "spots": ["大皇宫（玉佛寺·皇家建筑群）", "卧佛寺（46米卧佛·泰式按摩发源地）", "郑王庙（黎明寺·日落绝佳）"], "transport": "BTS+摆渡船", "cost": 400, "meals": {"breakfast": "酒店早餐", "lunch": "路边摊Pad Thai ¥20", "dinner": "考山路泰式料理 ¥60"}, "notes": "大皇宫500泰铢，必须穿长裤遮肩。卧佛寺按摩300泰铢/小时"},
                {"day": 2, "theme": "水上市场", "spots": ["丹嫩沙多水上市场（早市最热闹）", "美功铁道市场（火车穿市场奇观）", "河滨夜市（摩天轮·手信）"], "transport": "包车半日游", "cost": 350, "meals": {"breakfast": "水上市场船面¥15", "lunch": "水上市场烤虾¥40", "dinner": "河滨夜市海鲜¥80"}, "notes": "水上市场7-10点最热闹。美功市场火车时刻提前查"},
                {"day": 3, "theme": "购物与SPA", "spots": ["Siam商圈（Siam Paragon·Central World）", "Let's Relax SPA（泰式按摩2小时）", "恰图恰周末市场（15,000+摊位）"], "transport": "BTS", "cost": 500, "meals": {"breakfast": "酒店早餐", "lunch": "Siam美食广场¥30", "dinner": "建兴酒家咖喱蟹¥120"}, "notes": "恰图恰仅周末开放。SPA建议提前预约"},
                {"day": 4, "theme": "文化体验", "spots": ["吉姆·汤普森之家（泰丝博物馆）", "金山寺（318级台阶·360°全景）", "唐人街（耀华力路·燕窝街）"], "transport": "BTS+公交船", "cost": 300, "meals": {"breakfast": "街边粥¥10", "lunch": "唐人街点心¥40", "dinner": "唐人街海鲜¥80"}, "notes": "金山寺日落最美。唐人街晚上最热闹"},
                {"day": 5, "theme": "曼谷周边", "spots": ["大城遗址（世界文化遗产·半日游）", "暹罗博物馆（互动体验）", "考山路（背包客天堂·夜生活）"], "transport": "包车/火车", "cost": 350, "meals": {"breakfast": "酒店早餐", "lunch": "大城船面¥15", "dinner": "考山路街头美食¥50"}, "notes": "大城火车2小时/15泰铢超便宜。包车半日约1000泰铢"},
            ], "total_estimate": 1900, "currency": "CNY", "tips": ["BTS Rabbit卡充值方便，坐3次回本", "出租车必须by meter，否则被坑", "7-11是万能补给站，ATM取现手续费150泰铢", "SPA价格仅为国内1/3，推荐体验"]},
        "luxury": {"title": "曼谷5日奢华度假游", "overview": "半岛酒店·米其林泰餐·私人游艇·顶级SPA，泰式奢华极致体验",
            "days": [
                {"day": 1, "theme": "奢华抵达", "spots": ["曼谷半岛酒店（河景套房）", "私人游艇湄南河巡航", "Gaggan（亚洲50最佳餐厅）"], "transport": "机场豪华专车", "cost": 6000, "meals": {"breakfast": "机上", "lunch": "半岛酒店下午茶", "dinner": "Gaggan ¥3000"}, "notes": "半岛酒店下午茶全球闻名，需预约"},
                {"day": 2, "theme": "私人定制", "spots": ["大皇宫私人导览", "私人长尾船运河游", "Oasis Spa（总统套房SPA）"], "transport": "专车", "cost": 4000, "meals": {"breakfast": "半岛早餐", "lunch": "Sra Bua by Kiin Kiin ¥1500", "dinner": "Sühring（米其林二星）¥2000"}, "notes": "Oasis SPA是曼谷顶级SPA之一"},
                {"day": 3, "theme": "购物天堂", "spots": ["IconSiam（室内水上市场）", "Siam Paragon VIP购物", "Erb曼谷定制香水"], "transport": "专车", "cost": 3000, "meals": {"breakfast": "半岛早餐", "lunch": "IconSiam美食广场 ¥200", "dinner": "Le Normandie（米其林）¥2500"}, "notes": "IconSiam室内水上市场很独特"},
                {"day": 4, "theme": "华欣一日", "spots": ["华欣火车站（最古老最美）", "华欣葡萄园私人品酒", "华欣万豪度假村下午茶"], "transport": "专车2小时", "cost": 3000, "meals": {"breakfast": "半岛早餐", "lunch": "华欣海鲜餐厅 ¥500", "dinner": "回曼谷Bo.lan ¥1500"}, "notes": "华欣是皇室度假地，比普吉清静"},
                {"day": 5, "theme": "告别曼谷", "spots": ["半岛酒店SPA（退房前）", "暹罗博物馆VIP导览", "屋顶酒吧Three Eagles Bar"], "transport": "专车", "cost": 2000, "meals": {"breakfast": "半岛早餐", "lunch": "半岛酒店中餐 ¥800", "dinner": "Vertigo & Moon Bar ¥1500"}, "notes": "Banyan Tree顶楼酒吧360°曼谷夜景"},
            ], "total_estimate": 18000, "currency": "CNY", "tips": ["曼谷奢华酒店价格仅为国内1/2-1/3", "米其林餐厅性价比极高", "SPA一定要体验，世界顶级水平", "专车推荐用Grab Premium"]},
        "budget": {"title": "曼谷5日穷游攻略", "overview": "街头美食·夜市淘宝·寺庙免费，用最低预算感受微笑之国",
            "days": [
                {"day": 1, "theme": "免费寺庙", "spots": ["大皇宫（500泰铢必去）", "卧佛寺（200泰铢·按摩另付）", "郑王庙（50泰铢·摆渡4泰铢）"], "transport": "公交船16泰铢", "cost": 150, "meals": {"breakfast": "7-11饭团¥5", "lunch": "路边Pad Thai¥10", "dinner": "考山路街头美食¥20"}, "notes": "公交船比BTS便宜很多。卧佛寺按摩300泰铢超值"},
                {"day": 2, "theme": "水上市场", "spots": ["丹嫩沙多水上市场", "美功铁道市场（火车穿市场）"], "transport": "公共巴士+双条车", "cost": 100, "meals": {"breakfast": "船面¥5", "lunch": "水上市场烤虾¥30", "dinner": "7-11便当¥10"}, "notes": "公共交通比包车便宜5-10倍"},
                {"day": 3, "theme": "免费购物", "spots": ["恰图恰周末市场（免费逛）", "Siam商圈（免费逛·空调避暑）", "河滨夜市（免费逛·摩天轮另票）"], "transport": "BTS", "cost": 120, "meals": {"breakfast": "7-11", "lunch": "美食广场¥15", "dinner": "夜市小吃¥25"}, "notes": "恰图恰砍价先砍一半"},
                {"day": 4, "theme": "唐人街", "spots": ["金山寺（20泰铢）", "唐人街（免费逛）", "帕空花市（24小时·免费逛）"], "transport": "公交船", "cost": 80, "meals": {"breakfast": "街边粥¥5", "lunch": "唐人街鱼粥¥20", "dinner": "唐人街甜品¥10"}, "notes": "帕空花市凌晨最美"},
                {"day": 5, "theme": "大城省钱", "spots": ["大城遗址（火车15泰铢·2小时）"], "transport": "火车", "cost": 100, "meals": {"breakfast": "7-11", "lunch": "大城船面¥5", "dinner": "考山路¥20"}, "notes": "大城火车15泰铢！比包车便宜100倍"},
            ], "total_estimate": 550, "currency": "CNY", "tips": ["7-11是万能补给站，ATM取现手续费150泰铢", "公交船16泰铢横穿曼谷，比BTS便宜5倍", "路边摊最便宜最好吃，认准人多的", "双条车10-20泰铢，比出租便宜"]},
    },
}

# More destinations - London, Bali, Seoul, New York, Sydney, Dubai, Rome
_GENERIC_COMFORTABLE = {"title": "{dest}5日舒适游", "overview": "精选{dest}必去景点+地道美食+舒适住宿，深度体验当地文化",
    "days": [
        {"day": 1, "theme": "经典必游", "spots": ["{dest}标志性景点", "历史老城区", "中心广场", "地标观景台"], "transport": "公共交通", "cost": 800, "meals": {"breakfast": "酒店早餐", "lunch": "当地特色料理 ¥150", "dinner": "推荐餐厅 ¥300"}, "notes": "建议提前网上购票免排队"},
        {"day": 2, "theme": "文化探索", "spots": ["国家/城市博物馆", "艺术区/画廊", "历史遗迹", "特色街区"], "transport": "公共交通", "cost": 600, "meals": {"breakfast": "当地面包房", "lunch": "博物馆咖啡厅 ¥100", "dinner": "当地风味 ¥250"}, "notes": "博物馆通常有免费日，提前查询"},
        {"day": 3, "theme": "在地体验", "spots": ["当地市场", "特色体验（手工/烹饪课）", "自然景点/公园", "观景点日落"], "transport": "公共交通", "cost": 900, "meals": {"breakfast": "酒店早餐", "lunch": "市场美食 ¥80", "dinner": "网红餐厅 ¥350"}, "notes": "当地市场是感受文化的最佳场所"},
        {"day": 4, "theme": "周边一日游", "spots": ["周边古镇/自然景观", "世界文化遗产", "当地特色小镇", "特色美食体验"], "transport": "火车/大巴", "cost": 700, "meals": {"breakfast": "酒店早餐", "lunch": "当地料理 ¥120", "dinner": "返回后晚餐 ¥280"}, "notes": "提前查火车/大巴时刻表"},
        {"day": 5, "theme": "购物告别", "spots": ["购物中心/商业街", "手信/纪念品店", "最后打卡景点", "特色咖啡馆"], "transport": "公共交通", "cost": 500, "meals": {"breakfast": "酒店早餐", "lunch": "简餐 ¥80", "dinner": "告别大餐 ¥400"}, "notes": "退税手续需提前到机场办理"},
    ], "total_estimate": 3500, "currency": "CNY", "tips": ["买当地交通卡最方便", "提前下载离线地图", "随身携带护照方便退税", "注意当地小费文化"]}

_GENERIC_LUXURY = {"title": "{dest}5日奢华尊享游", "overview": "顶级酒店·私人导览·米其林盛宴，{dest}极致体验",
    "days": [
        {"day": 1, "theme": "奢华抵达", "spots": ["五星酒店入住", "私人接机", "欢迎晚宴"], "transport": "专车", "cost": 5000, "meals": {"breakfast": "机上", "lunch": "酒店餐厅", "dinner": "米其林推荐 ¥5000"}, "notes": "提前安排专车接机"},
        {"day": 2, "theme": "私人定制", "spots": ["私人导览·核心景点", "VIP通道免排队", "私人体验课程"], "transport": "专车", "cost": 4000, "meals": {"breakfast": "酒店早餐", "lunch": "高级餐厅 ¥2000", "dinner": "米其林星级 ¥6000"}, "notes": "私人导览需提前预约"},
        {"day": 3, "theme": "极致体验", "spots": ["VIP体验活动", "高端购物", "私人SPA"], "transport": "专车", "cost": 5000, "meals": {"breakfast": "酒店早餐", "lunch": "景观餐厅 ¥2500", "dinner": "主厨餐桌 ¥8000"}, "notes": "主厨餐桌至少提前1月预约"},
        {"day": 4, "theme": "周边私游", "spots": ["私人定制一日游", "专属交通", "独家体验"], "transport": "专车", "cost": 4000, "meals": {"breakfast": "酒店早餐", "lunch": "庄园餐厅 ¥3000", "dinner": "回城高级料理 ¥5000"}, "notes": "一日游路线可完全定制"},
        {"day": 5, "theme": "奢华告别", "spots": ["VIP购物顾问", "纪念品高级定制", "告别下午茶"], "transport": "专车送机", "cost": 3000, "meals": {"breakfast": "酒店早餐", "lunch": "酒店告别午餐 ¥2000", "dinner": "机场贵宾室"}, "notes": "退税手续酒店可代办"},
    ], "total_estimate": 21000, "currency": "CNY", "tips": ["顶级体验需提前2-4周预约", "奢华酒店通常含早餐和行政酒廊", "信用卡预订有保障", "专车比出租更舒适安全"]}

_GENERIC_BUDGET = {"title": "{dest}5日穷游攻略", "overview": "免费景点+平价美食+青旅住宿，用最低预算体验{dest}",
    "days": [
        {"day": 1, "theme": "免费打卡", "spots": ["{dest}标志景点（免费/低价）", "城市公园", "历史街区", "观景日落点"], "transport": "公共交通日票", "cost": 200, "meals": {"breakfast": "超市面包¥10", "lunch": "街头美食¥30", "dinner": "超市便当¥25"}, "notes": "买公共交通日票最省钱"},
        {"day": 2, "theme": "免费博物馆", "spots": ["免费博物馆", "免费画廊", "历史遗迹", "特色街区"], "transport": "步行+公交", "cost": 150, "meals": {"breakfast": "超市¥10", "lunch": "当地快餐¥25", "dinner": "超市¥20"}, "notes": "查询免费博物馆日"},
        {"day": 3, "theme": "在地体验", "spots": ["当地市场（免费逛）", "免费体验活动", "城市步道", "免费音乐/表演"], "transport": "步行", "cost": 100, "meals": {"breakfast": "超市¥10", "lunch": "市场小吃¥20", "dinner": "超市¥20"}, "notes": "市场是最省钱的美食来源"},
        {"day": 4, "theme": "周边省钱游", "spots": ["周边小镇/自然（公共交通）", "免费景点", "徒步路线"], "transport": "公共交通", "cost": 180, "meals": {"breakfast": "超市¥10", "lunch": "自带三明治", "dinner": "回城超市¥20"}, "notes": "自带午餐最省钱"},
        {"day": 5, "theme": "省钱告别", "spots": ["免费最后的景点", "手信在超市买", "最后的街头美食"], "transport": "公共交通", "cost": 100, "meals": {"breakfast": "超市¥10", "lunch": "街头美食¥20", "dinner": "超市¥15"}, "notes": "手信在当地超市买最便宜"},
    ], "total_estimate": 730, "currency": "CNY", "tips": ["住青旅/民宿每晚¥50-100", "超市是最省钱的选择", "买公共交通日票/周票", "免费景点很多，提前做攻略"]}


def get_client():
    global _client
    if not LLM_API_KEY:
        return None
    if _client is None:
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return _client


def is_llm_available():
    return bool(LLM_API_KEY)


def chat(prompt: str, system: str = "", intent: str = "") -> str:
    c = get_client()
    if not c:
        time.sleep(0.3 + random.random() * 0.4)
        return _mock(intent, prompt)
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    for attempt in range(3):
        try:
            r = c.chat.completions.create(model=LLM_MODEL, messages=msgs, temperature=0.7, max_tokens=4000)
            return r.choices[0].message.content or ""
        except Exception as e:
            if attempt == 2:
                return f"[LLM调用失败: {e}]，使用本地数据响应"
            time.sleep(2 ** attempt)


def chat_json(prompt: str, system: str = "", intent: str = "") -> dict:
    raw = chat(prompt, system, intent)
    try:
        if "```" in raw:
            m = re.search(r"```json\n?([\s\S]*?)\n?```", raw)
            if m:
                return json.loads(m.group(1))
        idx = raw.find("{")
        if idx >= 0:
            end = raw.rfind("}") + 1
            return json.loads(raw[idx:end])
        idx = raw.find("[")
        if idx >= 0:
            end = raw.rfind("]") + 1
            return json.loads(raw[idx:end])
        return {}
    except json.JSONDecodeError:
        return {}


def _detect_destination(text: str) -> str:
    for city in _DESTINATION_PLANS:
        if city in text:
            return city
    for city in ["伦敦", "巴厘岛", "首尔", "纽约", "悉尼", "迪拜", "罗马", "清迈", "大阪", "京都"]:
        if city in text:
            return city
    return ""


def _mock(intent: str, prompt: str = "") -> str:
    dest = _detect_destination(prompt)

    if intent in ("plan", "plan_luxury", "plan_budget"):
        style = {"plan_luxury": "luxury", "plan_budget": "budget"}.get(intent, "comfortable")
        if dest and dest in _DESTINATION_PLANS:
            return json.dumps(_DESTINATION_PLANS[dest][style], ensure_ascii=False)
        template = {"plan_luxury": _GENERIC_LUXURY, "plan_budget": _GENERIC_BUDGET}.get(intent, _GENERIC_COMFORTABLE)
        plan = json.loads(json.dumps(template).replace("{dest}", dest or "目的地"))
        return json.dumps(plan, ensure_ascii=False)

    if intent == "packing":
        if "冬" in prompt or "冬" in dest:
            items_extra = [{"item": "羽绒服/厚外套", "checked": False}, {"item": "保暖内衣", "checked": False}, {"item": "手套+围巾", "checked": False}, {"item": "暖宝宝", "checked": False}]
        else:
            items_extra = [{"item": "防晒霜SPF50+", "checked": False}, {"item": "遮阳帽", "checked": False}, {"item": "墨镜", "checked": False}, {"item": "防蚊液", "checked": False}]
        return json.dumps({"categories": [
            {"name": "证件", "items": [{"item": "护照+签证复印件", "checked": False}, {"item": "机票/车票行程单", "checked": False}, {"item": "酒店确认函", "checked": False}, {"item": "旅行保险单", "checked": False}, {"item": "证件照片2张(备用)", "checked": False}, {"item": "身份证(国内转机用)", "checked": False}]},
            {"name": "衣物", "items": [{"item": "轻便外套1件", "checked": False}, {"item": "T恤/衬衫4-5件", "checked": False}, {"item": "长裤2条+短裤1条", "checked": False}, {"item": "舒适步行鞋(已磨合)", "checked": False}, {"item": "拖鞋(旅馆/海滩用)", "checked": False}, {"item": "折叠伞/雨衣", "checked": False}] + items_extra},
            {"name": "电子设备", "items": [{"item": "充电宝(20000mAh以下)", "checked": False}, {"item": "转换插头(查目的地标准)", "checked": False}, {"item": "手机+充电线", "checked": False}, {"item": "相机/存储卡(可选)", "checked": False}, {"item": "耳机(飞机/地铁降噪)", "checked": False}]},
            {"name": "药品", "items": [{"item": "常备感冒药", "checked": False}, {"item": "肠胃药/止泻药", "checked": False}, {"item": "创可贴+消毒棉", "checked": False}, {"item": "晕车药(长途交通)", "checked": False}, {"item": "个人常用药(足量)", "checked": False}]},
            {"name": "其他", "items": [{"item": "当地交通卡(Suica/Oyster等)", "checked": False}, {"item": "当地货币现金(3-5天用量)", "checked": False}, {"item": "小塑料袋(垃圾分类/湿衣)", "checked": False}, {"item": "便携洗手液", "checked": False}, {"item": "折叠购物袋", "checked": False}, {"item": "旅行枕(长途飞行)", "checked": False}]},
        ]}, ensure_ascii=False)

    if intent == "guide":
        guides = {
            "东京": "# 东京旅行完整攻略\n\n## 签证须知\n- 中国公民需提前办理日本旅游签证\n- 单次签证有效期90天，停留15天\n- 三年多次签证需3年内去过日本\n- 五年多次签证要求年收50万+\n- 所需材料：护照、2寸白底照片2张、签证申请表、在职证明、近6个月银行流水（余额5万+）、机票酒店预订单\n\n## 当地交通\n- **Suica/Pasmo卡**：充值交通卡，地铁/公交/便利店通用，押金500日元可退\n- **东京Metro + 都营地铁**：覆盖全城，起步170日元，一日券600日元\n- **JR线**：山手线环线连接主要站点，中央线横贯东西\n- **出租车**：起步价约500日元，23:00-5:00加收20%深夜费\n- **新干线**：东京-大阪2.5小时，约14000日元\n\n## 美食推荐\n- **筑地场外市场**：寿司大¥4000、海胆饭¥3000、玉子烧¥200\n- **拉面横滨**：各家名店集中，一蘭¥1500、六厘舍¥1300\n- **居酒屋**：体验日式夜生活，人均3000-5000日元\n- **便利店**：Seven Premium系列品质超出预期\n- **唐吉诃德**：深夜打折，零食/药妆/日用品全有\n\n## 安全提示\n- 东京是全球最安全城市之一，深夜独行也安全\n- 防灾意识强，下载地震预警App「Yurekuru Call」\n- 紧急电话：110(警察) / 119(消防+急救)\n- 垃圾分类严格，随身带小塑料袋\n\n## 紧急联系方式\n- 中国驻日大使馆：03-3403-3388\n- 东京急救中心：03-3212-2323\n- 旅行者热线：050-3816-2787\n- 失物招领：03-3814-4151",
            "巴黎": "# 巴黎旅行完整攻略\n\n## 签证须知\n- 中国公民需办理申根签证\n- 申根签证可在26国通行\n- 有效期通常90天，停留最多90天\n- 所需材料：护照、照片、申请表、行程单、酒店预订单、保险、银行流水\n\n## 当地交通\n- **Metro**：16条线路覆盖全城，单程€2.15，10次票carnet€16.9\n- **RER**：郊区快线，去凡尔赛/机场必坐\n- **Navigo周票**：€30覆盖全区域，周一生效\n- **出租车**：起步€4，注意只坐正规出租\n\n## 美食推荐\n- **面包房**：羊角面包€1.2，法棍€1-1.5\n- **拉丁区**：学生区平价美食集中\n- **玛黑区**：L'As du Fallafel必吃\n- **米其林**：午间套餐性价比高，€40-80\n\n## 安全提示\n- 地铁和旅游景点注意防扒手\n- 18区、19区、20区夜间避免\n- 紧急电话：17(警察) / 15(急救) / 18(消防)\n\n## 紧急联系方式\n- 中国驻法大使馆：01 49 52 19 55\n- 旅游援助：01 45 62 10 40",
        }
        return guides.get(dest, guides.get("东京", "# 旅行攻略\n\n## 签证须知\n- 请查询目的地国家签证要求\n- 建议提前1个月办理\n\n## 当地交通\n- 建议购买当地交通卡\n- 出租车注意打表\n\n## 美食推荐\n- 尝试当地市场美食\n- 街头小吃往往最地道\n\n## 安全提示\n- 保管好护照和财物\n- 记录当地紧急电话\n\n## 紧急联系方式\n- 中国使馆电话请提前查询\n- 全球紧急求助：112"))

    if intent == "diary":
        if dest and dest in _DESTINATION_PLANS:
            plan = _DESTINATION_PLANS[dest]["comfortable"]
        else:
            plan = _DESTINATION_PLANS["东京"]["comfortable"]
        sections = []
        for day in plan["days"]:
            sections.append(f"## Day {day['day']} — {day['theme']}\n\n")
            spots_text = "、".join(day["spots"][:3])
            sections.append(f"今天的主题是{day['theme']}。{spots_text}，每一站都让人流连忘返。\n\n")
            if day.get("meals"):
                meals = day["meals"]
                if meals.get("dinner"):
                    sections.append(f"晚餐{meals['dinner']}，味蕾的满足让疲惫一扫而空。\n\n")
            sections.append(f"**今日花费**: ¥{day['cost']} | **最惊喜**: {day['spots'][-1]}\n\n")
        return f"# {dest or '东京'}旅行日记\n\n{''.join(sections)}\n---\n**总花费**: ¥{plan['total_estimate']} | **推荐指数**: ★★★★★"

    if intent == "budget_alloc":
        budget_val = 15000
        try:
            import re as _re
            m = _re.search(r'(\d+)', prompt)
            if m:
                budget_val = int(m.group(1))
        except Exception:
            pass
        return json.dumps({
            "total": budget_val, "currency": "CNY",
            "allocations": [
                {"category": "机票", "amount": int(budget_val * 0.28), "percent": 28, "color": "#6366f1", "icon": "✈️"},
                {"category": "住宿", "amount": int(budget_val * 0.25), "percent": 25, "color": "#8b5cf6", "icon": "🏨"},
                {"category": "餐饮", "amount": int(budget_val * 0.15), "percent": 15, "color": "#ec4899", "icon": "🍜"},
                {"category": "当地交通", "amount": int(budget_val * 0.10), "percent": 10, "color": "#f97316", "icon": "🚃"},
                {"category": "门票/体验", "amount": int(budget_val * 0.10), "percent": 10, "color": "#22c55e", "icon": "🎫"},
                {"category": "购物", "amount": int(budget_val * 0.07), "percent": 7, "color": "#06b6d4", "icon": "🛍️"},
                {"category": "应急储备", "amount": int(budget_val * 0.05), "percent": 5, "color": "#94a3b8", "icon": "🛡️"},
            ],
            "daily_budget": int(budget_val / 5),
            "daily_local": int(budget_val / 5 * 20),
            "tips": ["建议出发前在国内银行换汇，汇率优于机场", "每天记账，超出日预算时压缩餐饮开支", "应急储备不轻易动用，留作突发情况"],
        }, ensure_ascii=False)

    if intent == "settle":
        return json.dumps({
            "members": ["小明", "小红", "小华"],
            "expenses": [
                {"payer": "小明", "item": "酒店3晚", "amount": 2700, "category": "住宿"},
                {"payer": "小红", "item": "机票×3", "amount": 4500, "category": "交通"},
                {"payer": "小华", "item": "餐饮合计", "amount": 1800, "category": "餐饮"},
                {"payer": "小明", "item": "交通卡×3", "amount": 900, "category": "交通"},
                {"payer": "小红", "item": "门票合计", "amount": 1200, "category": "门票"},
                {"payer": "小华", "item": "购物", "amount": 600, "category": "购物"},
            ],
            "total": 11700,
            "per_person": 3900,
            "balances": [
                {"member": "小明", "paid": 3600, "share": 3900, "diff": -300},
                {"member": "小红", "paid": 5700, "share": 3900, "diff": 1800},
                {"member": "小华", "paid": 2400, "share": 3900, "diff": -1500},
            ],
            "settlements": [
                {"from": "小明", "to": "小红", "amount": 300, "reason": "小明付少补小红"},
                {"from": "小华", "to": "小红", "amount": 1500, "reason": "小华付少补小红"},
            ]
        }, ensure_ascii=False)

    if intent == "visa":
        visa_data = {
            "日本": {"visa_required": True, "visa_type": "短期滞在签证（旅游）", "validity": "90天", "stay_duration": "15天（单次）/30天（多次）", "processing_time": "5-7个工作日", "cost": "200元（单次）/400元（三年多次）/600元（五年多次）", "requirements": ["有效期6个月以上护照", "2寸白底照片2张", "签证申请表", "在职证明/营业执照副本", "近6个月银行流水（余额5万+）", "机票酒店预订单", "身份证正反面复印件"], "tips": ["建议提前1个月申请", "使馆可能电话核实，保持手机畅通", "首次申请建议找旅行社代办", "三年多次需3年内去过日本", "五年多次要求年收50万+"]},
            "法国": {"visa_required": True, "visa_type": "申根签证（旅游）", "validity": "90天", "stay_duration": "90天（180天内）", "processing_time": "10-15个工作日", "cost": "80欧元", "requirements": ["有效期6个月以上护照", "2寸白底照片2张", "申根签证申请表", "在职证明+公司营业执照", "近3个月银行流水（余额5万+）", "旅行保险（保额3万欧元+）", "机票+酒店+行程单", "身份证+户口本复印件"], "tips": ["建议提前2个月申请", "面试需法语或英语回答", "申根签证可在26国通行", "酒店可预订免费取消的"]},
            "泰国": {"visa_required": False, "visa_type": "落地签/免签", "validity": "N/A", "stay_duration": "30天（免签）/15天（落地签）", "processing_time": "免签无需", "cost": "免费（免签）/2000泰铢（落地签）", "requirements": ["有效期6个月以上护照", "返程机票", "酒店预订单", "2万泰铢等值货币（抽查）"], "tips": ["2024年起中国护照免签30天", "落地签排队长，建议提前电子签", "随身携带返程机票打印件"]},
        }
        data = visa_data.get(dest, visa_data["日本"])
        data["destination"] = dest or "日本"
        data["departure"] = "中国"
        return json.dumps(data, ensure_ascii=False)

    if intent == "sos":
        sos_data = {
            "日本": {"emergency": {"police": "110", "ambulance_fire": "119", "china_embassy": "03-3403-3388", "traveler_hotline": "050-3816-2787"}, "nearby": [{"name": "当地警察署", "type": "警察", "phone": "110", "tip": "可直接拨打"}, {"name": "东京医科大学病院", "type": "医院", "phone": "03-3342-6111", "tip": "24小时急诊"}, {"name": "中国驻日大使馆", "type": "使馆", "phone": "03-3403-3388", "tip": "工作日9:00-17:00"}], "phrases": [{"zh": "请帮帮我", "foreign": "助けてください", "pron": "Tasukete kudasai"}, {"zh": "请叫救护车", "foreign": "救急車を呼んでください", "pron": "Kyūkyūsha o yonde kudasai"}, {"zh": "我迷路了", "foreign": "道に迷いました", "pron": "Michi ni mayoimashita"}, {"zh": "我需要医生", "foreign": "医者が必要です", "pron": "Isha ga hitsuyō desu"}, {"zh": "有人偷了我的东西", "foreign": "盗まれました", "pron": "Nusumaremashita"}, {"zh": "我是中国公民", "foreign": "私は中国の市民です", "pron": "Watashi wa Chūgoku no shimin desu"}]},
            "法国": {"emergency": {"police": "17", "ambulance_fire": "15/18", "china_embassy": "01 49 52 19 55", "traveler_hotline": "112"}, "nearby": [{"name": "Commissariat", "type": "警察", "phone": "17", "tip": "全法通用"}, {"name": "SAMU", "type": "急救", "phone": "15", "tip": "24小时"}, {"name": "中国驻法大使馆", "type": "使馆", "phone": "01 49 52 19 55", "tip": "工作日"}], "phrases": [{"zh": "请帮帮我", "foreign": "Aidez-moi s'il vous plaît", "pron": "E-day mwah sil voo play"}, {"zh": "请叫救护车", "foreign": "Appelez une ambulance", "pron": "Ah-play oon ahm-boo-lahns"}, {"zh": "我迷路了", "foreign": "Je suis perdu(e)", "pron": "Zhuh swee pehr-doo"}, {"zh": "我是中国公民", "foreign": "Je suis citoyen(ne) chinois(e)", "pron": "Zhuh swee see-twah-yan shwa"}]},
        }
        data = sos_data.get(dest, sos_data["日本"])
        data["location"] = dest or "当前位置"
        data["country"] = dest or "当前国家"
        return json.dumps(data, ensure_ascii=False)

    if intent == "translate_scene":
        scenes = {
            ("餐厅点菜", "ja"): {"phrases": [
                {"zh": "请给我菜单", "foreign": "メニューをお願いします", "pron": "Menyū o onegaishimasu"},
                {"zh": "我要这个", "foreign": "これをお願いします", "pron": "Kore o onegaishimasu"},
                {"zh": "请结账", "foreign": "お会計をお願いします", "pron": "Okaikei o onegaishimasu"},
                {"zh": "有中文菜单吗", "foreign": "中国語のメニューはありますか", "pron": "Chūgokugo no menyū wa arimasu ka"},
                {"zh": "不要辣", "foreign": "辛くしないでください", "pron": "Karaku shinaide kudasai"},
                {"zh": "好吃！", "foreign": "おいしい！", "pron": "Oishī!"},
                {"zh": "还有座位吗", "foreign": "席はありますか", "pron": "Seki wa arimasu ka"},
                {"zh": "打包带走", "foreign": "持ち帰りでお願いします", "pron": "Mochikaeri de onegaishimasu"},
            ]},
            ("问路", "ja"): {"phrases": [
                {"zh": "请问XX怎么走", "foreign": "XXはどこですか", "pron": "XX wa doko desu ka"},
                {"zh": "离这里远吗", "foreign": "ここから遠いですか", "pron": "Koko kara tōi desu ka"},
                {"zh": "走过去可以吗", "foreign": "歩いて行けますか", "pron": "Aruite ikemasu ka"},
                {"zh": "最近的地铁站在哪", "foreign": "一番近い駅はどこですか", "pron": "Ichiban chikai eki wa doko desu ka"},
                {"zh": "需要多长时间", "foreign": "どのくらいかかりますか", "pron": "Dono kurai kakarimasu ka"},
                {"zh": "请画个地图给我", "foreign": "地図を書いてください", "pron": "Chizu o kaite kudasai"},
            ]},
            ("购物", "ja"): {"phrases": [
                {"zh": "可以退税吗", "foreign": "税返還できますか", "pron": "Zeihenkan dekimasu ka"},
                {"zh": "能便宜点吗", "foreign": "もう少し安くなりますか", "pron": "Mō sukoshi yasuku narimasu ka"},
                {"zh": "可以试穿吗", "foreign": "試着できますか", "pron": "Shichaku dekimasu ka"},
                {"zh": "有其他颜色吗", "foreign": "他の色はありますか", "pron": "Hoka no iro wa arimasu ka"},
                {"zh": "这个多少钱", "foreign": "これはいくらですか", "pron": "Kore wa ikura desu ka"},
                {"zh": "我用信用卡付", "foreign": "クレジットカードで払います", "pron": "Kurejitto kādo de haraimasu"},
            ]},
            ("入住酒店", "ja"): {"phrases": [
                {"zh": "我预订了房间", "foreign": "予約しています", "pron": "Yoyaku shiteimasu"},
                {"zh": "有WiFi吗", "foreign": "WiFiはありますか", "pron": "WiFi wa arimasu ka"},
                {"zh": "可以寄存行李吗", "foreign": "荷物を預かれますか", "pron": "Nimotsu o azukaremasu ka"},
                {"zh": "退房时间是几点", "foreign": "チェックアウトは何時ですか", "pron": "Chekku auto wa nanji desu ka"},
                {"zh": "可以延迟退房吗", "foreign": "レイトチェックアウトできますか", "pron": "Reito chekku auto dekimasu ka"},
                {"zh": "有洗衣服务吗", "foreign": "ランドリーサービスはありますか", "pron": "Randorī sābisu wa arimasu ka"},
            ]},
        }
        key = (prompt.split("场景：")[-1].split()[0] if "场景：" in prompt else "餐厅点菜", "ja")
        data = scenes.get(key, scenes[("餐厅点菜", "ja")])
        data["scene"] = key[0]
        data["language"] = key[1]
        return json.dumps(data, ensure_ascii=False)

    if intent == "compare":
        if dest and dest in _DESTINATION_PLANS:
            plans = []
            for style, label in [("budget", "穷游版"), ("comfortable", "舒适版"), ("luxury", "豪华版")]:
                p = _DESTINATION_PLANS[dest][style]
                plans.append({"name": label, "budget": p["total_estimate"], "style": p["overview"][:30], "highlights": [d["theme"] for d in p["days"][:3]], "days": len(p["days"]), "daily_cost": p["total_estimate"] // len(p["days"])})
            return json.dumps({"destination": dest, "plans": plans}, ensure_ascii=False)
        return json.dumps({"destination": dest or "目的地", "plans": [
            {"name": "穷游版", "budget": 5000, "style": "青旅+公共交通+街头美食", "highlights": ["体验最地道的本地生活", "街头美食探索", "免费景点打卡", "当地人市场"], "days": 5, "daily_cost": 1000},
            {"name": "舒适版", "budget": 15000, "style": "商务酒店+特色餐饮+深度体验", "highlights": ["品质住宿体验", "必去景点深度游", "特色美食体验", "文化沉浸"], "days": 5, "daily_cost": 3000},
            {"name": "豪华版", "budget": 35000, "style": "五星级+专车+米其林盛宴", "highlights": ["顶级酒店住宿", "米其林餐厅", "私人定制体验", "VIP免排队"], "days": 5, "daily_cost": 7000},
        ]}, ensure_ascii=False)

    if intent == "user_profile":
        return json.dumps({
            "budget_range": "1-2万", "travel_style": "文化体验+美食探索",
            "preferred_season": "春秋", "accommodation": "商务酒店/精品民宿",
            "dietary": "无特殊忌口，愿尝试当地料理", "past_trips": ["京都5日", "曼谷3日"],
            "interests": ["历史古迹", "当地美食", "摄影", "市集", "博物馆"],
        }, ensure_ascii=False)

    # ── Action-aware mock: detect booking/operation intents and generate tool calls ──
    lower_prompt = prompt.lower() if prompt else ""

    # Flight booking: "订" + ("航班" or "机票" or "票") or "book" or 飞/航班+订/买/预订
    wants_flight = any(kw in lower_prompt for kw in ["订航班", "订机票", "买机票", "买航班", "预订航班", "book flight", "book_flight"]) or \
                  ("订" in lower_prompt and any(kw in lower_prompt for kw in ["飞", "航班", "机票"])) or \
                  ("买" in lower_prompt and any(kw in lower_prompt for kw in ["飞", "航班", "机票"]))
    if wants_flight:
        # Parse route info
        dep = "北京"
        arr = "东京"
        for city in ["东京","巴黎","曼谷","首尔","伦敦","纽约","悉尼","迪拜","罗马","巴厘岛"]:
            if city in prompt:
                arr = city; break
        passenger = "张三"
        for m in re.finditer(r'乘客[是为：:\s]*(\S+)', prompt):
            passenger = m.group(1)
        date = ""
        for m in re.finditer(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', prompt):
            date = m.group(1)
        return f'[call:flight] {{"departure":"{dep}","destination":"{arr}","date":"{date}"}}\n\n根据查询结果，我为您预订最便宜的航班：\n\n[call:book_flight] {{"airline":"春秋 9C8515","departure":"{dep}","arrival":"{arr}","date":"{date}","passenger":"{passenger}","price":1500,"seat":"无偏好","meal":"标准"}}\n\n✅ 已为您完成预订！航班信息：春秋9C8515 {dep}→{arr} | ¥1,500 | 乘客：{passenger}'

    # Hotel booking
    wants_hotel = any(kw in lower_prompt for kw in ["订酒店", "预订酒店", "book hotel", "book_hotel", "订房", "预订房", "订住宿"]) or \
                  ("订" in lower_prompt and "酒店" in lower_prompt) or \
                  ("预订" in lower_prompt and "酒店" in lower_prompt)
    if wants_hotel:
        dest = "东京"
        for city in ["东京","巴黎","曼谷","首尔","伦敦","纽约","悉尼","迪拜","罗马","巴厘岛"]:
            if city in prompt: dest = city; break
        guest = get_profile().get("name", "张三") if get_profile() else "张三"
        return f'[call:hotel] {{"destination":"{dest}","budget_per_night":800,"style":"舒适"}}\n\n根据查询结果，为您推荐并预订：\n\n[call:book_hotel] {{"name":"新宿格兰贝尔酒店","city":"{dest}","check_in":"待确认","check_out":"待确认","guest":"{guest}","room_type":"标准间","price_per_night":600,"nights":1,"guests":1}}\n\n✅ 酒店预订成功！新宿格兰贝尔酒店 | 标准间 | ¥600/晚'

    if "加入行程" in lower_prompt or "添加景点" in lower_prompt or "add_spot" in lower_prompt or ("加" in lower_prompt and "行程" in lower_prompt):
        spot = "景点"
        city = ""
        for m in re.finditer(r'把(.+?)加入', prompt):
            spot = m.group(1)
        for city_name in ["东京","巴黎","曼谷","首尔","伦敦","纽约","悉尼","迪拜","罗马","巴厘岛"]:
            if city_name in prompt: city = city_name; break
        return f'[call:add_spot] {{"name":"{spot}","city":"{city}","note":"用户添加"}}\n\n✅ 已将「{spot}」添加到您的行程！'

    if "提醒" in lower_prompt or "别忘了" in lower_prompt or "add_reminder" in lower_prompt:
        text = prompt.replace("提醒我","").replace("别忘了","").strip()[:50]
        date = ""
        for m in re.finditer(r'(\d{1,2}月\d{1,2}号?|\d{4}[-/]\d{1,2}[-/]\d{1,2})', prompt):
            date = m.group(1)
        return f'[call:add_reminder] {{"text":"{text}","date":"{date}","type":"旅行提醒"}}\n\n✅ 已添加提醒：{text}（{date}）'

    if "收藏" in lower_prompt and ("翻译" in lower_prompt or "短语" in lower_prompt):
        return "请告诉我您想收藏的中文和对应翻译，我来帮您收藏到短语本。"

    if "收藏" in lower_prompt:
        name = prompt.replace("收藏","").strip()[:20]
        return f'[call:add_spot] {{"name":"{name}","city":"","note":"用户收藏"}}\n\n✅ 已收藏「{name}」！'

    return "好的，我来帮您规划！请告诉我您的具体需求，包括目的地、出行时间、天数和预算。"
