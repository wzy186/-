import json
from tools.base import BaseTool


class TranslateTool(BaseTool):
    name = "translate"
    description = "多语言翻译（8种语言+场景+发音）"

    PHRASES = {
        "ja": {
            "你好": {"t": "こんにちは", "p": "Konnichiwa"},
            "谢谢": {"t": "ありがとうございます", "p": "Arigatō gozaimasu"},
            "对不起": {"t": "すみません", "p": "Sumimasen"},
            "多少钱": {"t": "いくらですか", "p": "Ikura desu ka"},
            "在哪里": {"t": "どこですか", "p": "Doko desu ka"},
            "请帮帮我": {"t": "助けてください", "p": "Tasukete kudasai"},
            "我要这个": {"t": "これをお願いします", "p": "Kore o onegaishimasu"},
            "请结账": {"t": "お会計をお願いします", "p": "Okaikei o onegaishimasu"},
            "有中文菜单吗": {"t": "中国語のメニューはありますか", "p": "Chūgokugo no menyū wa arimasu ka"},
            "卫生间在哪": {"t": "トイレはどこですか", "p": "Toire wa doko desu ka"},
            "太好了": {"t": "すごい！", "p": "Sugoi!"},
            "再见": {"t": "さようなら", "p": "Sayōnara"},
            "好吃": {"t": "おいしい！", "p": "Oishī!"},
            "请等一下": {"t": "ちょっと待ってください", "p": "Chotto matte kudasai"},
            "我迷路了": {"t": "道に迷いました", "p": "Michi ni mayoimashita"},
        },
        "en": {
            "你好": {"t": "Hello", "p": "Heh-LOH"},
            "谢谢": {"t": "Thank you", "p": "THANGK yoo"},
            "对不起": {"t": "Excuse me", "p": "ek-SKYOOZ mee"},
            "多少钱": {"t": "How much?", "p": "how MUCH"},
            "在哪里": {"t": "Where is?", "p": "WHERE iz"},
            "请帮帮我": {"t": "Please help me", "p": "pleez HELP mee"},
            "我要这个": {"t": "I'll take this one", "p": "yl tayk this wun"},
            "请结账": {"t": "Check, please", "p": "CHEK pleez"},
            "卫生间在哪": {"t": "Where is the restroom?", "p": "WHERE iz thuh REST-room"},
            "太好了": {"t": "Great!", "p": "GRAYT"},
            "再见": {"t": "Goodbye", "p": "guud-BY"},
            "好吃": {"t": "Delicious!", "p": "deh-LISH-us"},
            "请等一下": {"t": "Just a moment, please", "p": "just uh MOH-ment pleez"},
            "我迷路了": {"t": "I'm lost", "p": "ym LAWST"},
        },
        "fr": {
            "你好": {"t": "Bonjour", "p": "bohn-ZHOOR"},
            "谢谢": {"t": "Merci", "p": "mehr-SEE"},
            "对不起": {"t": "Pardon", "p": "par-DOHN"},
            "多少钱": {"t": "Combien?", "p": "kohn-BYEHN"},
            "在哪里": {"t": "Où est?", "p": "oo EH"},
            "请帮帮我": {"t": "Aidez-moi, s'il vous plaît", "p": "E-day mwah, seel voo PLEH"},
            "我要这个": {"t": "Je vais prendre celui-ci", "p": "Zhuh veh prend suh-LWEE-see"},
            "请结账": {"t": "L'addition, s'il vous plaît", "p": "Lah-dee-SYOHN, seel voo PLEH"},
            "卫生间在哪": {"t": "Où sont les toilettes?", "p": "oo SOHN lay twah-LET"},
            "太好了": {"t": "C'est super!", "p": "SEH soo-PEHR"},
            "再见": {"t": "Au revoir", "p": "OH ruh-VWAHR"},
            "好吃": {"t": "C'est délicieux!", "p": "SEH day-lee-SYUH"},
        },
        "th": {
            "你好": {"t": "สวัสดี", "p": "Sa-wat-dee"},
            "谢谢": {"t": "ขอบคุณ", "p": "Khop-khun"},
            "对不起": {"t": "ขอโทษ", "p": "Kho-thot"},
            "多少钱": {"t": "เท่าไหร่", "p": "Tao-rai"},
            "在哪里": {"t": "ที่ไหน", "p": "Thee-nai"},
            "请帮帮我": {"t": "ช่วยด้วย", "p": "Chuay duay"},
            "我要这个": {"t": "เอาอันนี้", "p": "Ao an-nee"},
            "好吃": {"t": "อร่อย", "p": "A-roy"},
            "太辣了": {"t": "เผ็ดเกินไป", "p": "Pet gern-pai"},
            "便宜点": {"t": "ลดหน่อย", "p": "Lot noi"},
        },
        "ko": {
            "你好": {"t": "안녕하세요", "p": "An-nyeong-ha-se-yo"},
            "谢谢": {"t": "감사합니다", "p": "Gam-sa-ham-ni-da"},
            "对不起": {"t": "죄송합니다", "p": "Jwe-song-ham-ni-da"},
            "多少钱": {"t": "얼마예요?", "p": "Eol-ma-ye-yo"},
            "在哪里": {"t": "어디예요?", "p": "Eo-di-ye-yo"},
            "好吃": {"t": "맛있어요!", "p": "Ma-sis-seo-yo"},
            "便宜点": {"t": "깎아주세요", "p": "Kka-kka-ju-se-yo"},
        },
        "it": {
            "你好": {"t": "Ciao", "p": "CHOW"},
            "谢谢": {"t": "Grazie", "p": "GRAH-tsyeh"},
            "对不起": {"t": "Mi scusi", "p": "mee SKOO-zee"},
            "多少钱": {"t": "Quanto costa?", "p": "KWAHN-toh KOH-stah"},
            "好吃": {"t": "Buonissimo!", "p": "bwoh-NEES-see-moh"},
        },
        "es": {
            "你好": {"t": "Hola", "p": "OH-lah"},
            "谢谢": {"t": "Gracias", "p": "GRAH-syahs"},
            "对不起": {"t": "Perdón", "p": "pehr-DOHN"},
            "多少钱": {"t": "¿Cuánto cuesta?", "p": "KWAHN-toh KWEH-stah"},
            "好吃": {"t": "¡Delicioso!", "p": "deh-lee-SYOH-soh"},
        },
        "de": {
            "你好": {"t": "Guten Tag", "p": "GOO-ten tahk"},
            "谢谢": {"t": "Danke", "p": "DAHN-keh"},
            "对不起": {"t": "Entschuldigung", "p": "ENT-shool-dee-goong"},
            "多少钱": {"t": "Wie viel kostet das?", "p": "vee feel KOH-stet dahs"},
            "好吃": {"t": "Lecker!", "p": "LEH-kehr"},
        },
    }

    LANG_MAP = {
        "日语": "ja", "英语": "en", "法语": "fr", "泰语": "th", "韩语": "ko",
        "意大利语": "it", "西班牙语": "es", "德语": "de",
        "日本": "ja", "英国": "en", "法国": "fr", "泰国": "th", "韩国": "ko",
        "意大利": "it", "西班牙": "es", "德国": "de",
    }

    def run(self, args: dict) -> str:
        text = args.get("text", "")
        target = args.get("target", "ja")
        scene = args.get("scene", "")
        target = self.LANG_MAP.get(target, target)
        phrases = self.PHRASES.get(target, self.PHRASES["en"])

        if text and text in phrases:
            entry = phrases[text]
            return json.dumps({
                "text": text, "translated": entry["t"], "pronunciation": entry["p"],
                "language": target, "scene": scene,
            }, ensure_ascii=False)

        all_phrases = []
        for zh, data in phrases.items():
            all_phrases.append({"zh": zh, "foreign": data["t"], "pron": data["p"]})

        return json.dumps({
            "language": target, "scene": scene or "通用",
            "phrases": all_phrases,
            "tip": f"支持8种语言翻译，说出具体句子可获得精确翻译",
        }, ensure_ascii=False)
