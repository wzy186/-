#!/usr/bin/env python3
# 生成商铺封面 + 分类图标 SVG(占位用,无真实图片资源时的视觉占位)
import os

OUT = "/Users/didi/doudshengsheng/doudshengsheng-web/public/covers"
os.makedirs(OUT, exist_ok=True)

# (id, name, 渐变c1, 渐变c2, emoji)
shops = [
    (1, "奶茶", "#FF9A8B", "#FF6A88", "🧋"),
    (2, "火锅", "#FF6B6B", "#EE0979", "🍲"),
    (3, "KTV", "#667eea", "#764ba2", "🎤"),
    (4, "美甲", "#f093fb", "#f5576c", "💅"),
    (5, "酒店", "#4facfe", "#00f2fe", "🏨"),
    (6, "烤肉", "#fa709a", "#fee140", "🥩"),
    (7, "电影", "#30cfd0", "#330867", "🎬"),
    (8, "理发", "#a8edea", "#fed6e3", "💈"),
]

def cover(emoji, c1, c2):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="400" height="240" viewBox="0 0 400 240">
  <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/>
  </linearGradient></defs>
  <rect width="400" height="240" fill="url(#g)"/>
  <circle cx="320" cy="40" r="60" fill="rgba(255,255,255,0.15)"/>
  <circle cx="60" cy="210" r="40" fill="rgba(255,255,255,0.1)"/>
  <text x="200" y="135" font-size="80" text-anchor="middle" fill="rgba(255,255,255,0.92)">{emoji}</text>
</svg>'''

for sid, _, c1, c2, em in shops:
    with open(f"{OUT}/shop_{sid}.svg", "w") as f:
        f.write(cover(em, c1, c2))

# 分类图标(圆形带 emoji)
cats = [
    (1, "美食", "🍜", "#FF6A88"),
    (2, "娱乐", "🎮", "#764ba2"),
    (3, "丽人", "💄", "#f5576c"),
    (4, "服务", "💈", "#43cea2"),
    (5, "酒店", "🏨", "#4facfe"),
    (6, "秒杀", "⚡", "#FF5A36"),
    (7, "红包", "🧧", "#EE0979"),
    (8, "签到", "📅", "#667eea"),
]
for cid, name, em, c in cats:
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
  <circle cx="40" cy="40" r="36" fill="{c}" opacity="0.12"/>
  <text x="40" y="52" font-size="34" text-anchor="middle">{em}</text>
</svg>'''
    with open(f"{OUT}/cat_{cid}.svg", "w") as f:
        f.write(svg)

print(f"生成 {len(shops)} 个商铺封面 + {len(cats)} 个分类图标 到 {OUT}")
