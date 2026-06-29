import streamlit as st
import json
import os
import sys
import base64

sys.path.insert(0, os.path.dirname(__file__))

from core.agent import process, resume
from core.llm import chat, chat_json, is_llm_available
from core.prompts import *
from core.rag import search, index_destinations, get_context
from core.memory import get_profile, update_profile, get_session, get_recent_sessions, clear_session, get_trip_history
from tools.weather import WeatherTool
from tools.flight import FlightTool
from tools.exchange import ExchangeTool
from tools.hotel import HotelTool
from tools.amap import AmapTool
from tools.budget_tool import BudgetTool
from utils.storage import (
    register_user, login_user, logout_user, get_current_user,
    get_nickname, set_nickname, get_avatar, set_avatar,
    add_booking, remove_booking, get_bookings, update_booking,
    add_favorite, remove_favorite, get_favorites,
    add_itinerary_spot, remove_itinerary_spot, get_itinerary_spots, clear_itinerary_spots,
    save_phrase, get_saved_phrases, remove_saved_phrase,
    set_visa_check, get_visa_check,
    save_note, get_note, get_all_notes,
    add_reminder, get_reminders, toggle_reminder, remove_reminder,
    add_trip_log, get_trip_log, update_trip_log,
    set_wallet, get_wallet,
    get_stats, _load, _STORAGE_FILE,
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

st.set_page_config(page_title="TravelMate — AI智能出行助手", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

# ═══════════════ CUSTOM CSS ═══════════════
st.markdown("""
<style>
/* ── Global ── */
.main .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1400px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
section[data-testid="stSidebar"] .sidebar-content { padding: 0.5rem 0; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0 !important; }

/* ── Nav button style ── */
.nav-btn > button { background: transparent !important; border: none !important; color: #94a3b8 !important; text-align: left !important; padding: 0.5rem 0.8rem !important; border-radius: 10px !important; font-size: 0.88rem !important; transition: all 0.2s !important; }
.nav-btn > button:hover { background: rgba(255,255,255,0.08) !important; color: #e2e8f0 !important; }
.nav-btn-active > button { background: linear-gradient(135deg, #3b82f6, #6366f1) !important; color: white !important; font-weight: 600 !important; box-shadow: 0 4px 12px rgba(59,130,246,0.3) !important; }

/* ── Cards ── */
.card { background: white; border-radius: 14px; padding: 1.2rem; margin-bottom: 0.8rem; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }

/* ── Hero ── */
.hero-section { background: linear-gradient(135deg, #1e3a5f 0%, #2d1b69 50%, #1e3a5f 100%); border-radius: 16px; padding: 2rem 2.5rem; color: white; margin-bottom: 1.5rem; position: relative; overflow: hidden; }

/* ── Badges ── */
.badge { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 0.7rem; font-weight: 600; }
.badge-green { background: #dcfce7; color: #166534; }
.badge-yellow { background: #fef3c7; color: #92400e; }
.badge-red { background: #fee2e2; color: #991b1b; }
.badge-blue { background: #dbeafe; color: #1e40af; }

/* ── Action result ── */
.action-done { background: linear-gradient(135deg, #f0fdf4, #dcfce7); border-left: 4px solid #22c55e; padding: 10px 14px; border-radius: 8px; margin: 6px 0; }
.tool-chain { background: linear-gradient(135deg, #fffbeb, #fef3c7); padding: 6px 10px; border-radius: 8px; font-size: 0.85em; border-left: 3px solid #f59e0b; }

/* ── Image cards ── */
.city-card { border-radius: 14px; overflow: hidden; border: 1px solid #e2e8f0; background: white; transition: transform 0.2s, box-shadow 0.2s; }
.city-card:hover { transform: translateY(-4px); box-shadow: 0 12px 28px rgba(0,0,0,0.12); }
.city-card img { width: 100%; height: 160px; object-fit: cover; display: block; }
.city-card .city-info { padding: 0.8rem 1rem; }
.city-card .city-name { font-weight: 700; font-size: 1.1rem; color: #1e293b; }
.city-card .city-desc { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }

/* ── Stat ── */
.stat-box { text-align: center; padding: 0.8rem; border-radius: 12px; background: linear-gradient(135deg, #f8fafc, #eef2ff); border: 1px solid #e0e7ff; }
.stat-box .stat-val { font-size: 1.5rem; font-weight: 700; color: #1e293b; }
.stat-box .stat-lbl { font-size: 0.7rem; color: #64748b; }

/* ── Progress bar ── */
.stProgress > div > div > div { background: linear-gradient(90deg, #3b82f6, #6366f1); border-radius: 4px; }

/* ── Login page ── */
.login-container { max-width: 400px; margin: 4rem auto; }

/* ── Route visualization ── */
.route-step { display: flex; align-items: flex-start; gap: 0.8rem; padding: 0.6rem 0; border-bottom: 1px solid #f1f5f9; }
.route-step:last-child { border-bottom: none; }
.route-dot { width: 12px; height: 12px; border-radius: 50%; background: #3b82f6; flex-shrink: 0; margin-top: 4px; }
.route-dot.end { background: #22c55e; }
.route-line { width: 2px; background: #3b82f6; height: 20px; margin-left: 5px; }

/* ── Divider ── */
.divider { height: 1px; background: linear-gradient(90deg, transparent, #e2e8f0, transparent); margin: 1.5rem 0; }

/* ── Metric ── */
.stMetric > div { background: linear-gradient(135deg, #f8fafc, #eef2ff); border-radius: 10px; padding: 8px 12px; border: 1px solid #e0e7ff; }
</style>
""", unsafe_allow_html=True)

# ═══════════════ SESSION STATE ═══════════════
for key, val in {
    "chat_history": [], "current_trip": None, "session_id": "default",
    "packing_list": None, "guide_text": None, "diary_text": None,
    "budget_data": None, "settle_data": None, "visa_data": None,
    "sos_data": None, "translate_data": None, "compare_data": None,
    "search_results": None, "weather_data": None, "expense_list": [],
    "flight_search_result": None, "hotel_search_result": None,
    "attraction_search_result": None, "route_result": None,
    "nearby_result": None, "settle_expenses_list": [],
    "current_page": "主页", "logged_in": False, "username": "",
    "login_tab": "login", "pending_interrupt": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Check persisted login
if not st.session_state.logged_in:
    persisted = get_current_user()
    if persisted:
        st.session_state.logged_in = True
        st.session_state.username = persisted

try:
    index_destinations()
except Exception:
    pass

# ═══════════════ HELPERS ═══════════════
CITY_DATA = {
    "东京": {"key": "tokyo", "desc": "樱花与科技交织的都城", "tags": "文化·美食·购物"},
    "巴黎": {"key": "paris", "desc": "浪漫之都，艺术殿堂", "tags": "浪漫·艺术·美食"},
    "曼谷": {"key": "bangkok", "desc": "热带风情，寺庙之城", "tags": "美食·寺庙·夜市"},
    "巴厘岛": {"key": "bali", "desc": "海岛天堂，心灵净土", "tags": "海滩·瑜伽·潜水"},
    "伦敦": {"key": "london", "desc": "英伦经典，皇室风范", "tags": "博物馆·下午茶·购物"},
    "首尔": {"key": "seoul", "desc": "韩流发源地，古今交融", "tags": "韩流·美食·购物"},
    "纽约": {"key": "newyork", "desc": "不夜城，世界之巅", "tags": "百老汇·博物馆·购物"},
    "悉尼": {"key": "sydney", "desc": "港湾明珠，自然与都市", "tags": "海滩·歌剧·潜水"},
    "迪拜": {"key": "dubai", "desc": "沙漠奇迹，奢华之都", "tags": "奢华·购物·沙漠"},
    "罗马": {"key": "rome", "desc": "永恒之城，历史长廊", "tags": "古迹·美食·艺术"},
}

def _get_city_img(city_name: str) -> str | None:
    info = CITY_DATA.get(city_name)
    if not info:
        return None
    path = os.path.join(STATIC_DIR, f"{info['key']}.svg")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/svg+xml;base64,{data}"
    return None

def _nav_to(page: str):
    st.session_state.current_page = page
    st.rerun()

def _section_title(icon, text):
    st.markdown(f'<div style="font-size:1.3rem;font-weight:700;color:#1e293b;margin-bottom:0.8rem;display:flex;align-items:center;gap:0.5rem;">{icon} {text}</div>', unsafe_allow_html=True)

def _back_button(target_page="主页"):
    if st.button("← 返回", key=f"back_to_{target_page}"):
        _nav_to(target_page)

def _stat_box(val, lbl):
    st.markdown(f'<div class="stat-box"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

def _badge(text, color="blue"):
    return f'<span class="badge badge-{color}">{text}</span>'

# ═══════════════ LOGIN / REGISTER PAGE ═══════════════
def page_auth():
    st.markdown("""
    <div style="text-align:center; padding:3rem 0 1rem 0;">
        <div style="font-size:4rem;">✈️</div>
        <h1 style="color:#1e293b; margin:0.5rem 0;">TravelMate</h1>
        <p style="color:#64748b; font-size:1.1rem;">AI 智能出行助手 · 让每一次旅行都轻松无忧</p>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["🔐 登录", "📝 注册"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="输入用户名")
            password = st.text_input("密码", type="password", placeholder="输入密码")
            if st.form_submit_button("登录", type="primary", use_container_width=True):
                if not username or not password:
                    st.error("请输入用户名和密码")
                else:
                    result = login_user(username, password)
                    if result["success"]:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.current_page = "主页"
                        st.rerun()
                    else:
                        st.error(result["error"])

    with tab_register:
        with st.form("register_form"):
            reg_user = st.text_input("用户名", placeholder="设置用户名（英文/数字）")
            reg_nick = st.text_input("昵称", placeholder="你的旅行昵称")
            reg_pwd = st.text_input("密码", type="password", placeholder="设置密码")
            reg_pwd2 = st.text_input("确认密码", type="password", placeholder="再次输入密码")
            if st.form_submit_button("注册", type="primary", use_container_width=True):
                if not reg_user or not reg_pwd:
                    st.error("请填写用户名和密码")
                elif reg_pwd != reg_pwd2:
                    st.error("两次密码不一致")
                elif len(reg_pwd) < 4:
                    st.error("密码至少4位")
                else:
                    result = register_user(reg_user, reg_pwd, reg_nick or reg_user)
                    if result["success"]:
                        login_user(reg_user, reg_pwd)
                        st.session_state.logged_in = True
                        st.session_state.username = reg_user
                        st.session_state.current_page = "主页"
                        st.rerun()
                    else:
                        st.error(result["error"])

    # Feature preview
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#64748b; font-size:0.85rem; padding:1rem 0;">
        🤖 AI行程规划 &nbsp;|&nbsp; ✈️ 航班酒店预订 &nbsp;|&nbsp; 💰 预算管理 &nbsp;|&nbsp; 🌐 翻译助手 &nbsp;|&nbsp; 🗺️ 地图导航 &nbsp;|&nbsp; 🆘 紧急求助
    </div>
    """, unsafe_allow_html=True)

# ═══════════════ SIDEBAR (only when logged in) ═══════════════
NAV_GROUPS = [
    ("发现", ["🏠 主页", "🗺️ 行程规划", "📊 方案对比"]),
    ("出行服务", ["✈️ 航班", "🏨 酒店", "📍 景点", "🗺️ 地图", "🌤️ 天气"]),
    ("工具箱", ["💰 预算", "💱 汇率", "🌐 翻译", "📋 清单", "🛂 签证", "👥 AA分摊"]),
    ("记录", ["📖 攻略", "📝 日记", "🔍 知识库", "🆘 SOS"]),
    ("我的", ["👤 个人主页", "💬 AI助手", "⭐ 收藏", "🔔 提醒", "⚙️ 设置"]),
]

def render_sidebar():
    with st.sidebar:
        # Brand
        nickname = get_nickname(st.session_state.username)
        avatar = get_avatar(st.session_state.username)
        st.markdown(f"""
        <div style="text-align:center; padding:0.8rem 0 0.3rem 0;">
            <div style="font-size:1.8rem;">{avatar}</div>
            <div style="font-size:0.95rem; font-weight:600; color:#f1f5f9;">{nickname}</div>
        </div>
        """, unsafe_allow_html=True)

        # Status
        statuses = []
        statuses.append('<span class="badge badge-green">AI在线</span>' if is_llm_available() else '<span class="badge badge-yellow">Mock</span>')
        statuses.append('<span class="badge badge-green">地图</span>' if os.getenv("AMAP_API_KEY") else '<span class="badge badge-yellow">地图Mock</span>')
        st.markdown(f'<div style="text-align:center;margin:0.2rem 0;">{"&nbsp;".join(statuses)}</div>', unsafe_allow_html=True)

        # Navigation
        for group_name, pages in NAV_GROUPS:
            st.markdown(f'<div style="color:#64748b;font-size:0.65rem;text-transform:uppercase;letter-spacing:2px;margin:0.8rem 0 0.2rem 0.8rem;font-weight:600;">{group_name}</div>', unsafe_allow_html=True)
            for page_label in pages:
                icon = page_label.split(" ")[0]
                name = page_label.split(" ", 1)[1]
                is_active = st.session_state.current_page == name
                css_class = "nav-btn-active" if is_active else "nav-btn"
                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                if st.button(f"{icon} {name}", key=f"nav_{name}", use_container_width=True):
                    _nav_to(name)
                st.markdown('</div>', unsafe_allow_html=True)

        # Stats
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        stats = get_stats(st.session_state.username)
        c1, c2 = st.columns(2)
        c1.metric("旅行", f"{stats['total_trips']}次")
        c2.metric("预订", f"{stats['total_bookings']}项")

        # Quick reminders
        reminders = [r for r in get_reminders(st.session_state.username) if not r.get("done")]
        if reminders:
            st.markdown("**🔔 待办**")
            for r in reminders[:3]:
                st.caption(f"• {r.get('text','')} ({r.get('date','')})")

        # Logout
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        if st.button("🚪 退出登录", use_container_width=True):
            logout_user()
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

# ═══════════════ PAGE: 主页 ═══════════════
def page_home():
    nickname = get_nickname(st.session_state.username)
    st.markdown(f"""
    <div class="hero-section">
        <h1>🌍 你好，{nickname}</h1>
        <p>告诉我你想去哪，我来帮你搞定一切 — 从规划到预订，一气呵成</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick actions
    qa_cols = st.columns(4)
    qa_items = [
        ("🗺️", "规划行程", "AI 智能规划", "行程规划"),
        ("✈️", "查航班", "比价+预订", "航班"),
        ("🏨", "订酒店", "星级筛选", "酒店"),
        ("💬", "AI 助手", "对话式出行", "AI助手"),
    ]
    for i, (icon, title, desc, page) in enumerate(qa_items):
        with qa_cols[i]:
            if st.button(f"{icon} {title}\n{desc}", key=f"home_qa_{page}", use_container_width=True):
                _nav_to(page)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Stats
    stats = get_stats(st.session_state.username)
    stat_cols = st.columns(5)
    for i, (val, lbl) in enumerate([
        (f"{stats['total_trips']}", "🌍 旅行"),
        (f"{stats['total_bookings']}", "🎫 预订"),
        (f"{stats['total_favorites']}", "⭐ 收藏"),
        (f"{stats['total_phrases']}", "🌐 短语"),
        (f"{stats['total_reminders']}", "🔔 待办"),
    ]):
        with stat_cols[i]:
            _stat_box(val, lbl)

    # Hot destinations with images
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    _section_title("🔥", "热门目的地")
    dest_cols = st.columns(5)
    hot_cities = ["东京", "巴黎", "曼谷", "巴厘岛", "伦敦"]
    for i, city in enumerate(hot_cities):
        with dest_cols[i]:
            info = CITY_DATA.get(city, {})
            img = _get_city_img(city)
            img_html = f'<img src="{img}" style="width:100%;height:120px;object-fit:cover;border-radius:10px;">' if img else ''
            st.markdown(f"""
            <div class="city-card">
                {img_html}
                <div class="city-info">
                    <div class="city-name">{city}</div>
                    <div class="city-desc">{info.get('desc','')}</div>
                    <div style="font-size:0.7rem;color:#6366f1;margin-top:0.2rem;">{info.get('tags','')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"探索{city}", key=f"exp_{city}", use_container_width=True):
                st.session_state.current_page = "景点"
                st.rerun()

    # More destinations row
    dest_cols2 = st.columns(5)
    more_cities = ["首尔", "纽约", "悉尼", "迪拜", "罗马"]
    for i, city in enumerate(more_cities):
        with dest_cols2[i]:
            info = CITY_DATA.get(city, {})
            img = _get_city_img(city)
            img_html = f'<img src="{img}" style="width:100%;height:120px;object-fit:cover;border-radius:10px;">' if img else ''
            st.markdown(f"""
            <div class="city-card">
                {img_html}
                <div class="city-info">
                    <div class="city-name">{city}</div>
                    <div class="city-desc">{info.get('desc','')}</div>
                    <div style="font-size:0.7rem;color:#6366f1;margin-top:0.2rem;">{info.get('tags','')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"探索{city}", key=f"exp2_{city}", use_container_width=True):
                st.session_state.current_page = "景点"
                st.rerun()

    # Current trip
    trip = st.session_state.current_trip
    if trip:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        _section_title("🗺️", "当前行程")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("行程", trip.get("title", "未规划")[:20])
        c2.metric("天数", f"{len(trip.get('days', []))}天")
        c3.metric("预估", f"¥{trip.get('total_estimate', 0):,}")
        c4.metric("景点", f"{sum(len(d.get('spots',[])) for d in trip.get('days',[]))}个")
        for day in trip.get("days", [])[:3]:
            st.caption(f"Day {day['day']} {day.get('theme','')}: {', '.join(day.get('spots',[])[:3])}")

    # Upcoming bookings
    bookings = get_bookings(username=st.session_state.username)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    _section_title("🎫", "即将出行")
    if bookings:
        upcoming = [b for b in bookings if b.get("status") != "已取消"][-4:]
        for b in upcoming:
            icon = {"flight": "✈️", "hotel": "🏨"}.get(b.get("type", ""), "📋")
            with st.expander(f"{icon} {b.get('name','')} — {b.get('status','')}"):
                for k, v in b.items():
                    if k not in ("id",) and v:
                        st.caption(f"**{k}**: {v}")
                c1, c2 = st.columns(2)
                if c1.button("❌ 取消", key=f"cancel_{b['id']}"):
                    update_booking(b["id"], {"status": "已取消"})
                    st.rerun()
                if c2.button("🗑️ 删除", key=f"del_{b['id']}"):
                    remove_booking(b["id"])
                    st.rerun()
    else:
        st.caption("暂无预订。在 AI 助手中说「帮我订票」即可预订！")

    # Reminders
    reminders = [r for r in get_reminders(st.session_state.username) if not r.get("done")]
    if reminders:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        _section_title("🔔", "待办提醒")
        for r in reminders[:5]:
            c1, c2, c3 = st.columns([8, 1, 1])
            c1.markdown(f"• 📌 {r.get('text','')} — {r.get('date','')} ({r.get('type','')})")
            if c2.button("✅", key=f"done_rem_{r['id']}"):
                toggle_reminder(r["id"])
                st.rerun()
            if c3.button("🗑️", key=f"del_rem_{r['id']}"):
                remove_reminder(r["id"])
                st.rerun()

    # Budget
    bd = st.session_state.budget_data
    if bd:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        _section_title("💰", "预算速览")
        total = bd.get("total", 0)
        spent = sum(e["amount"] for e in st.session_state.expense_list)
        pct = spent / total * 100 if total else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("预算", f"¥{total:,}")
        c2.metric("已花", f"¥{spent:,}")
        c3.metric("剩余", f"¥{total-spent:,}", f"{100-pct:.0f}%")

# ═══════════════ PAGE: 个人主页 ═══════════════
def page_profile():
    _back_button()
    _section_title("👤", "个人主页")
    profile = get_profile(st.session_state.username)

    # Avatar + nickname
    c1, c2 = st.columns([1, 3])
    with c1:
        cur_avatar = get_avatar(st.session_state.username)
        st.markdown(f'<div style="text-align:center;font-size:4rem;">{cur_avatar}</div>', unsafe_allow_html=True)
        new_avatar = st.selectbox("头像", ["🧑‍✈️", "👩‍💻", "🧑‍🎨", "🏄", "🧗", "📷", "🎵", "🌏"], index=["🧑‍✈️","👩‍💻","🧑‍🎨","🏄","🧗","📷","🎵","🌏"].index(cur_avatar) if cur_avatar in ["🧑‍✈️","👩‍💻","🧑‍🎨","🏄","🧗","📷","🎵","🌏"] else 0, key="pf_avatar")
        if new_avatar != cur_avatar:
            set_avatar(new_avatar, st.session_state.username)
    with c2:
        cur_nick = get_nickname(st.session_state.username)
        new_nick = st.text_input("昵称", cur_nick, key="pf_name")
        if new_nick != cur_nick:
            set_nickname(new_nick, st.session_state.username)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Profile details
    c1, c2 = st.columns(2)
    with c1:
        budget_range = st.text_input("预算范围", profile.get("budget_range", "1-2万"), key="pf_budget")
        travel_style = st.text_input("旅行风格", profile.get("travel_style", "文化体验+美食"), key="pf_style")
        preferred_season = st.text_input("偏好季节", profile.get("preferred_season", "春秋"), key="pf_season")
    with c2:
        accommodation = st.text_input("住宿偏好", profile.get("accommodation", "商务酒店/民宿"), key="pf_acc")
        dietary = st.text_input("饮食偏好", profile.get("dietary", "无特殊忌口"), key="pf_diet")
        interests = st.text_input("兴趣", profile.get("interests", "寺庙,美食,摄影"), key="pf_int")

    avoid = st.text_input("不喜欢", profile.get("avoid", ""), key="pf_avoid")
    passport = st.text_input("护照类型", profile.get("passport", "中国护照"), key="pf_passport")
    frequent_flyer = st.text_input("常旅客号", profile.get("frequent_flyer", ""), key="pf_ff", placeholder="如: 国航CAxxxxx")

    if st.button("💾 保存个人资料", type="primary", use_container_width=True):
        update_profile({
            "budget_range": budget_range, "travel_style": travel_style, "preferred_season": preferred_season,
            "accommodation": accommodation, "dietary": dietary, "interests": interests, "avoid": avoid,
            "passport": passport, "frequent_flyer": frequent_flyer,
        }, st.session_state.username)
        st.success("资料已保存！AI 将自动参考您的偏好")

    # Stats
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    _section_title("📊", "旅行统计")
    trip_log = get_trip_log(st.session_state.username)
    bookings = get_bookings(username=st.session_state.username)
    favorites = get_favorites(username=st.session_state.username)
    phrases = get_saved_phrases(st.session_state.username)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌍 旅行次数", len(trip_log))
    c2.metric("🎫 总预订", len(bookings))
    c3.metric("⭐ 总收藏", len(favorites))
    c4.metric("🌐 收藏短语", len(phrases))

    if trip_log:
        dests = [t.get("destination", "") for t in trip_log if t.get("destination")]
        if dests:
            st.markdown(f"**去过的目的地**: {', '.join(set(dests))}")

    # Wallet
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    _section_title("💱", "我的钱包")
    wallet = get_wallet(st.session_state.username)
    currencies = ["CNY", "JPY", "USD", "EUR", "THB", "GBP", "KRW", "AUD"]
    c1, c2 = st.columns(2)
    with c1:
        for curr in currencies[:4]:
            amt = wallet.get(curr, 0)
            new_amt = st.number_input(f"{curr}", 0.0, 1000000.0, float(amt), key=f"wallet_{curr}")
            if new_amt != amt: set_wallet(curr, new_amt, st.session_state.username)
    with c2:
        for curr in currencies[4:]:
            amt = wallet.get(curr, 0)
            new_amt = st.number_input(f"{curr}", 0.0, 1000000.0, float(amt), key=f"wallet_{curr}")
            if new_amt != amt: set_wallet(curr, new_amt, st.session_state.username)

    from tools.exchange import ExchangeTool as ET
    et = ET()
    total_cny = 0
    for curr, amt in get_wallet(st.session_state.username).items():
        if curr == "CNY": total_cny += amt
        elif amt > 0:
            try:
                r = json.loads(et.run({"amount": amt, "from": curr, "to": "CNY"}))
                total_cny += r.get("result", 0)
            except: pass
    st.metric("钱包总计(≈CNY)", f"¥{total_cny:,.0f}")

    # Trip history
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    _section_title("🌍", "旅行记录")
    if trip_log:
        for t in reversed(trip_log[-10:]):
            with st.expander(f"📍 {t.get('destination','?')} · {t.get('days','?')}天 · ¥{t.get('budget',0):,}"):
                st.caption(f"风格: {t.get('style','')} | 创建: {t.get('created_at','')[:10]}")
    else:
        st.caption("暂无旅行记录，规划第一个行程吧！")

# ═══════════════ PAGE: AI 助手 ═══════════════
def _handle_agent_result(result):
    """Common handler for agent results: display reply, actions, trace, HITL."""
    reply = result["reply"]
    actions = result.get("actions", [])

    # HITL: show confirmation dialog if interrupted
    if result.get("interrupted"):
        interrupt_data = result.get("interrupt_data", {})
        st.warning(interrupt_data.get("message", "需要确认操作"))
        c1, c2 = st.columns(2)
        if c1.button("✅ 确认执行", type="primary", use_container_width=True):
            resume_result = resume(st.session_state.session_id, "yes")
            st.session_state.pending_interrupt = None
            for act in resume_result.get("actions", []):
                if isinstance(act.get("result"), dict) and act["result"].get("success"):
                    st.markdown(f'<div class="action-done">{act["result"]["message"]}</div>', unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "assistant", "content": resume_result["reply"]})
            st.rerun()
        if c2.button("❌ 取消操作", use_container_width=True):
            resume_result = resume(st.session_state.session_id, "no")
            st.session_state.pending_interrupt = None
            st.info(resume_result.get("reply", "操作已取消"))
            st.session_state.chat_history.append({"role": "assistant", "content": resume_result.get("reply", "操作已取消")})
            st.rerun()
        return reply  # Don't finalize yet

    # Show action results
    if actions:
        for act in actions:
            act_data = act.get("result", {})
            if isinstance(act_data, dict) and act_data.get("success"):
                st.markdown(f'<div class="action-done">{act_data["message"]}</div>', unsafe_allow_html=True)

    # Show reply
    try:
        parsed = json.loads(reply)
        st.json(parsed)
        reply_display = f"```json\n{reply}\n```"
    except:
        reply_display = reply
        st.markdown(reply)

    # Tool chain
    if result.get("tool_calls"):
        tc_str = " → ".join(f"🔧 {tc['tool']}" for tc in result["tool_calls"])
        st.markdown(f'<div class="tool-chain">Agent 调用链: {tc_str}</div>', unsafe_allow_html=True)

    # Thinking
    if result.get("thinking"):
        with st.expander("💭 思考过程"):
            for step in result["thinking"]:
                st.caption(step)

    # Tracing
    if result.get("trace"):
        with st.expander("📊 执行追踪"):
            for t in result["trace"]:
                node = t.get("node", "")
                ms = t.get("elapsed_ms", 0)
                changes = list(t.get("state_changes", {}).keys())
                st.caption(f"**{node}** — {ms}ms | 变更: {', '.join(changes) if changes else '无'}")

    return reply_display


def page_chat():
    _back_button()
    _section_title("💬", "AI 出行助手")
    st.caption("我能帮你查信息、订票、预订酒店、加景点到行程、设提醒。试试「帮我订北京到东京最便宜的航班」")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if "✅" in content and ("预订成功" in content or "已添加" in content or "已收藏" in content or "已保存" in content):
                st.markdown(f'<div class="action-done">{content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(content)

    # Pending HITL confirmation
    if st.session_state.pending_interrupt:
        interrupt_data = st.session_state.pending_interrupt
        st.warning(interrupt_data.get("message", "需要确认操作"))
        c1, c2 = st.columns(2)
        if c1.button("✅ 确认执行", type="primary", use_container_width=True):
            resume_result = resume(st.session_state.session_id, "yes")
            st.session_state.pending_interrupt = None
            for act in resume_result.get("actions", []):
                if isinstance(act.get("result"), dict) and act["result"].get("success"):
                    st.markdown(f'<div class="action-done">{act["result"]["message"]}</div>', unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "assistant", "content": resume_result["reply"]})
            st.rerun()
        if c2.button("❌ 取消操作", use_container_width=True):
            resume_result = resume(st.session_state.session_id, "no")
            st.session_state.pending_interrupt = None
            st.session_state.chat_history.append({"role": "assistant", "content": resume_result.get("reply", "操作已取消")})
            st.rerun()

    if prompt := st.chat_input("问我任何出行问题，如「帮我订7月15号北京到东京的航班」"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("AI 思考中..."):
                result = process(prompt, st.session_state.session_id)
                reply_display = _handle_agent_result(result)
                if result.get("interrupted"):
                    st.session_state.pending_interrupt = result.get("interrupt_data")
                    st.session_state.chat_history.append({"role": "assistant", "content": "⏳ 等待确认操作..."})
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": reply_display})

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("**🚀 快捷操作**")
    qa = [
        ("✈️ 订最便宜航班", "帮我订北京到东京最便宜的航班，乘客用我的名字"),
        ("🏨 订新宿酒店", "帮我预订东京新宿附近每晚800元以内的酒店，5晚"),
        ("🌤️ 查天气", "东京这周天气怎么样？"),
        ("🔔 设提醒", "提醒我7月15号去办日本签证"),
    ]
    cols = st.columns(4)
    for i, (label, q) in enumerate(qa):
        if cols[i].button(label, key=f"qq_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("AI 思考中..."):
                    result = process(q, st.session_state.session_id)
                    reply_display = _handle_agent_result(result)
                    if result.get("interrupted"):
                        st.session_state.pending_interrupt = result.get("interrupt_data")
                        st.session_state.chat_history.append({"role": "assistant", "content": "⏳ 等待确认操作..."})
                    else:
                        st.session_state.chat_history.append({"role": "assistant", "content": reply_display})
            st.rerun()

# ═══════════════ PAGE: 行程规划 ═══════════════
def page_plan():
    _back_button()
    _section_title("🗺️", "行程规划")
    c1, c2, c3 = st.columns(3)
    with c1:
        destination = st.text_input("目的地", "东京", key="plan_dest")
        days = st.number_input("天数", 1, 30, 5, key="plan_days")
    with c2:
        budget = st.number_input("预算(CNY)", 1000, 100000, 15000, key="plan_budget")
        style = st.selectbox("风格", ["文化体验", "美食之旅", "自然风光", "购物休闲", "亲子游"], key="plan_style")
    with c3:
        season = st.selectbox("季节", ["春", "夏", "秋", "冬"], key="plan_season")
        plan_type = st.selectbox("档次", ["舒适版", "穷游版", "豪华版"], key="plan_type")

    if st.button("🚀 生成行程", type="primary", use_container_width=True):
        with st.spinner("AI 规划中..."):
            intent = {"穷游版": "plan_budget", "豪华版": "plan_luxury"}.get(plan_type, "plan")
            request_str = f"去{destination}{days}天，预算{budget}元，{style}，{season}季出行"
            context = get_context(destination)
            result = chat_json(PLAN_PROMPT.format(request=request_str, context=context), intent=intent)
            if not result:
                try: result = json.loads(chat("", intent=intent))
                except: result = {}
            st.session_state.current_trip = result
            add_trip_log({"destination": destination, "days": days, "budget": budget, "style": style, "plan_type": plan_type}, st.session_state.username)

    trip = st.session_state.current_trip
    if trip:
        # Show destination image
        img = _get_city_img(destination)
        if img:
            st.markdown(f'<img src="{img}" style="width:100%;max-height:200px;object-fit:cover;border-radius:12px;margin-bottom:1rem;">', unsafe_allow_html=True)

        st.subheader(trip.get("title", "行程方案"))
        st.markdown(trip.get("overview", ""))
        total = trip.get("total_estimate", 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("总预算", f"¥{budget:,}")
        c2.metric("预估花费", f"¥{total:,}")
        c3.metric("日均", f"¥{total // max(days, 1):,}")

        for day in trip.get("days", []):
            with st.expander(f"Day {day['day']} — {day.get('theme','')}", expanded=True):
                for si, spot in enumerate(day.get("spots", [])):
                    c1, c2 = st.columns([10, 1])
                    c1.markdown(f"- 📍 {spot}")
                    if c2.button("✏️", key=f"rm_d{day['day']}_s{si}"):
                        day["spots"].pop(si)
                        st.rerun()
                new_spot = st.text_input("➕ 添加景点", "", key=f"add_d{day['day']}", placeholder="回车添加")
                if new_spot:
                    day.setdefault("spots", []).append(new_spot)
                    st.rerun()
                note_key = f"day{day['day']}_note"
                existing = get_note(note_key, st.session_state.username)
                new_note = st.text_input("📝 备注", existing, key=f"note_d{day['day']}")
                if new_note != existing and new_note:
                    save_note(note_key, new_note, st.session_state.username)
                meals = day.get("meals", {})
                if meals:
                    mc1, mc2, mc3 = st.columns(3)
                    if meals.get("breakfast"): mc1.caption(f"🌅 {meals['breakfast']}")
                    if meals.get("lunch"): mc2.caption(f"☀️ {meals['lunch']}")
                    if meals.get("dinner"): mc3.caption(f"🌙 {meals['dinner']}")
                c1, c2, c3 = st.columns(3)
                c1.metric("🚃", day.get("transport", ""))
                c2.metric("💰", f"¥{day.get('cost', 0):,}")
                if day.get("notes"): c3.caption(f"💡 {day['notes']}")

        extra = get_itinerary_spots(st.session_state.username)
        if extra:
            st.subheader("📌 补充景点")
            for i, s in enumerate(extra):
                c1, c2 = st.columns([10, 1])
                c1.markdown(f"- 📍 {s.get('name','')} ({s.get('city','')}) {s.get('note','')}")
                if c2.button("❌", key=f"rm_extra_{i}"):
                    remove_itinerary_spot(i, st.session_state.username)
                    st.rerun()
            if st.button("🗑️ 清空补充景点"):
                clear_itinerary_spots(st.session_state.username)
                st.rerun()

        for tip in trip.get("tips", []): st.caption(f"💡 {tip}")

        c1, c2, c3 = st.columns(3)
        if c1.button("💾 导出MD"):
            os.makedirs("saved_trips", exist_ok=True)
            md = f"# {trip.get('title','')}\n\n"
            for day in trip.get("days", []):
                md += f"## Day {day['day']} — {day.get('theme','')}\n"
                for s in day.get("spots", []): md += f"- {s}\n"
                md += f"🚃 {day.get('transport','')} | 💰 ¥{day.get('cost',0)}\n\n"
            with open(f"saved_trips/{destination}_{days}days.md", "w", encoding="utf-8") as f: f.write(md)
            st.success("已导出!")
        if c2.button("📋 导出JSON"):
            os.makedirs("saved_trips", exist_ok=True)
            with open(f"saved_trips/{destination}_{days}days.json", "w", encoding="utf-8") as f:
                json.dump(trip, f, ensure_ascii=False, indent=2)
            st.success("已导出!")
        if c3.button("⭐ 收藏行程"):
            add_favorite({"type": "行程", "name": trip.get("title", ""), "detail": f"{destination} {days}天 ¥{total:,}"}, st.session_state.username)
            st.success("已收藏!")

# ═══════════════ PAGE: 航班 ═══════════════
def page_flight():
    _back_button()
    _section_title("✈️", "航班查询 & 预订")
    c1, c2 = st.columns(2)
    with c1: f_origin = st.text_input("出发", "北京", key="f_origin")
    with c2: f_dest = st.text_input("到达", "东京", key="f_dest")
    f_date = st.text_input("日期", "", key="f_date", placeholder="如 2025-07-15")
    if st.button("✈️ 查询", type="primary", use_container_width=True):
        st.session_state.flight_search_result = json.loads(FlightTool().run({"departure": f_origin, "destination": f_dest, "date": f_date}))
    fd = st.session_state.flight_search_result
    if fd:
        cheapest = fd.get("cheapest", {})
        fastest = fd.get("fastest", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("最低价", f"¥{cheapest.get('price','N/A')}", cheapest.get("airline",""))
        c2.metric("最快", fastest.get("duration","N/A"), fastest.get("airline",""))
        c3.metric("航班数", str(fd.get("count",0)))

        for f in fd.get("flights", []):
            badges = []
            if f.get("price") == cheapest.get("price"): badges.append("💰最低")
            if f.get("airline") == fastest.get("airline"): badges.append("⚡最快")
            if "廉航" in f.get("type",""): badges.append("🏷️廉航")
            with st.expander(f"✈️ {f['airline']} ¥{f['price']:,} {' '.join(badges)}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("时间", f"{f['depart']}→{f['arrive']}")
                c2.metric("时长", f["duration"])
                c3.metric("价格", f"¥{f['price']:,}")
                c1, c2, c3, c4 = st.columns(4)
                c1.caption(f"机型: {f.get('aircraft','')}")
                c2.caption(f"餐食: {f.get('meal','')}")
                c3.caption(f"行李: {f.get('baggage','')}")
                c4.caption(f"WiFi: {f.get('wifi','')}")
                st.divider()
                st.markdown("**🎫 预订此航班**")
                with st.form(f"book_f_{f['airline'].replace(' ','')}"):
                    bc1, bc2 = st.columns(2)
                    passenger = bc1.text_input("乘客姓名", get_nickname(st.session_state.username), key=f"fp_{f['airline']}")
                    seat = bc2.selectbox("座位", ["靠窗", "靠走道", "无偏好"], key=f"fs_{f['airline']}")
                    meal = st.selectbox("餐食", ["标准", "素食", "清真", "无偏好"], key=f"fm_{f['airline']}")
                    col_book, col_fav = st.columns(2)
                    if col_book.form_submit_button("🎫 确认预订"):
                        if passenger:
                            bid = add_booking({
                                "type": "flight", "name": f"{f['airline']} {f_origin}→{f_dest}",
                                "airline": f["airline"], "route": f"{f_origin}→{f_dest}",
                                "date": f_date or "待确认", "depart": f["depart"], "arrive": f["arrive"],
                                "price": f["price"], "passenger": passenger, "seat": seat, "meal": meal, "status": "已确认",
                            }, st.session_state.username)
                            st.success(f"✅ 预订成功！订单号: {bid}")
                        else:
                            st.error("请输入乘客姓名")
                    if col_fav.form_submit_button("⭐ 收藏"):
                        add_favorite({"type": "航班", "name": f"{f['airline']}", "detail": f"{f_origin}→{f_dest} ¥{f['price']:,}"}, st.session_state.username)
                        st.success("已收藏!")

# ═══════════════ PAGE: 酒店 ═══════════════
def page_hotel():
    _back_button()
    _section_title("🏨", "酒店 & 预订")
    h_dest = st.text_input("目的地", "东京", key="h_dest")
    h_budget = st.number_input("每晚预算(CNY)", 100, 10000, 800, key="h_budget")
    h_style = st.selectbox("风格", ["舒适", "穷游", "豪华"], key="h_style")
    if st.button("🏨 搜索", type="primary", use_container_width=True):
        st.session_state.hotel_search_result = json.loads(HotelTool().run({"destination": h_dest, "budget_per_night": h_budget, "style": h_style}))
    hd = st.session_state.hotel_search_result
    if hd:
        for hotel in hd.get("hotels", []):
            with st.expander(f"⭐{hotel.get('stars',0)} {hotel['name']} ¥{hotel['price']}/晚"):
                c1, c2, c3 = st.columns(3)
                c1.metric("评分", f"⭐{hotel.get('rating','')}")
                c2.metric("价格", f"¥{hotel['price']}/晚")
                c3.metric("类型", hotel.get("type",""))
                if hotel.get("amenities"): st.caption(f"设施: {' | '.join(hotel['amenities'])}")
                if hotel.get("review_highlight"): st.info(f"💬 {hotel['review_highlight']}")
                st.divider()
                st.markdown("**🏨 预订**")
                with st.form(f"book_h_{hotel['name'].replace(' ','')}"):
                    bc1, bc2 = st.columns(2)
                    guest = bc1.text_input("入住人", get_nickname(st.session_state.username), key=f"hg_{hotel['name']}")
                    room = bc2.selectbox("房型", ["标准间", "大床房", "双床房", "套房"], key=f"hrt_{hotel['name']}")
                    bc1, bc2 = st.columns(2)
                    check_in = bc1.text_input("入住", "", key=f"hci_{hotel['name']}", placeholder="2025-07-15")
                    check_out = bc2.text_input("退房", "", key=f"hco_{hotel['name']}", placeholder="2025-07-20")
                    guests = st.number_input("人数", 1, 4, 2, key=f"hgn_{hotel['name']}")
                    col_book, col_fav, col_add = st.columns(3)
                    if col_book.form_submit_button("🏨 预订"):
                        if guest:
                            nights = 1
                            try:
                                from datetime import datetime as dt
                                if check_in and check_out:
                                    nights = (dt.strptime(check_out,"%Y-%m-%d") - dt.strptime(check_in,"%Y-%m-%d")).days
                            except: pass
                            total_cost = hotel["price"] * max(nights, 1)
                            bid = add_booking({"type":"hotel","name":hotel["name"],"city":h_dest,"room_type":room,
                                "check_in":check_in or "待确认","check_out":check_out or "待确认","nights":nights,
                                "price_per_night":hotel["price"],"total":total_cost,"guest":guest,"guests":guests,"status":"已确认"}, st.session_state.username)
                            st.success(f"✅ 预订成功！订单号:{bid} 总价¥{total_cost:,}")
                        else:
                            st.error("请输入入住人")
                    if col_fav.form_submit_button("⭐ 收藏"):
                        add_favorite({"type":"酒店","name":hotel["name"],"detail":f"¥{hotel['price']}/晚 ⭐{hotel.get('rating','')}"}, st.session_state.username)
                        st.success("已收藏!")
                    if col_add.form_submit_button("📍 加入行程"):
                        add_itinerary_spot({"name":f"🏨 {hotel['name']}","city":h_dest,"note":f"¥{hotel['price']}/晚"}, st.session_state.username)
                        st.success("已加入行程!")
        if hd.get("area_guide"):
            guide = hd["area_guide"]
            if guide.get("推荐区域"): st.markdown("**推荐**: " + " | ".join(guide["推荐区域"]))
            for w in guide.get("避坑", []): st.warning(f"⚠️ {w}")

# ═══════════════ PAGE: 景点 ═══════════════
def page_attraction():
    _back_button()
    _section_title("📍", "景点推荐")
    a_dest = st.text_input("城市", "东京", key="a_dest")
    # City image
    img = _get_city_img(a_dest)
    if st.button("📍 搜索", type="primary", use_container_width=True):
        from tools.attraction import AttractionTool
        st.session_state.attraction_search_result = json.loads(AttractionTool().run({"city": a_dest, "query": f"{a_dest} 景点"}))
    if img:
        st.markdown(f'<img src="{img}" style="width:100%;max-height:180px;object-fit:cover;border-radius:12px;margin-bottom:1rem;">', unsafe_allow_html=True)
    ad = st.session_state.attraction_search_result
    if ad:
        for attr in ad.get("attractions", []):
            with st.expander(f"📍 {attr.get('name','')} ⭐{attr.get('rating','')} {attr.get('price','')}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("类型", attr.get("type",""))
                c2.metric("时长", attr.get("duration",""))
                c3.metric("最佳时段", attr.get("best_time",""))
                if attr.get("highlight"): st.info(f"💡 {attr['highlight']}")
                c1, c2, c3 = st.columns(3)
                if c1.button("📍加入行程", key=f"add_{attr.get('name','')}"):
                    add_itinerary_spot({"name": attr.get("name",""), "city": a_dest, "note": f"⭐{attr.get('rating','')} {attr.get('price','')}"}, st.session_state.username)
                    st.success("已加入!")
                if c2.button("⭐收藏", key=f"fav_{attr.get('name','')}"):
                    add_favorite({"type":"景点","name":attr.get("name",""),"detail":f"{attr.get('type','')} ⭐{attr.get('rating','')}"}, st.session_state.username)
                    st.success("已收藏!")
                if c3.button("🗺️查路线", key=f"rt_{attr.get('name','')}"):
                    st.session_state.current_page = "地图"
                    st.rerun()

# ═══════════════ PAGE: 地图 (ENHANCED) ═══════════════
def page_map():
    _back_button()
    _section_title("🗺️", "地图导航")
    map_action = st.selectbox("功能", ["路线规划", "周边搜索", "地理编码", "行政区划"], key="map_action")

    if map_action == "路线规划":
        c1, c2 = st.columns(2)
        with c1: origin = st.text_input("起点", "新宿站", key="map_o")
        with c2: dest = st.text_input("终点", "浅草寺", key="map_d")
        mode = st.selectbox("方式", ["驾车", "公交", "步行", "骑行"], key="map_m")
        mode_icons = {"驾车": "🚗", "公交": "🚇", "步行": "🚶", "骑行": "🚴"}

        if st.button("🔍 规划", type="primary", use_container_width=True):
            st.session_state.route_result = json.loads(AmapTool("route").run({"origin_name":origin,"destination_name":dest,"mode":mode}))
        rr = st.session_state.route_result
        if rr:
            rd = rr.get("data", rr)

            # Visual route card
            st.markdown(f"""
            <div class="card">
                <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;">
                    <div style="text-align:center;">
                        <div style="font-size:2rem;">{mode_icons.get(mode,'🚗')}</div>
                        <div style="font-size:0.7rem;color:#64748b;">{mode}</div>
                    </div>
                    <div style="flex:1;">
                        <div style="display:flex;align-items:center;gap:0.5rem;">
                            <span class="badge badge-blue">{origin}</span>
                            <span style="color:#94a3b8;">→</span>
                            <span class="badge badge-green">{dest}</span>
                        </div>
                    </div>
                </div>
                <div style="display:flex;gap:2rem;">
                    <div style="text-align:center;">
                        <div style="font-size:1.4rem;font-weight:700;color:#1e293b;">{rd.get('distance','')}</div>
                        <div style="font-size:0.7rem;color:#64748b;">总距离</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:1.4rem;font-weight:700;color:#1e293b;">{rd.get('duration','')}</div>
                        <div style="font-size:0.7rem;color:#64748b;">预计时间</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Route steps visualization
            if rd.get("steps"):
                st.markdown("**📋 路线详情**")
                steps_html = ""
                for idx, s in enumerate(rd["steps"]):
                    dot_class = "end" if idx == len(rd["steps"]) - 1 else ""
                    steps_html += f"""
                    <div class="route-step">
                        <div>
                            <div class="route-dot {dot_class}"></div>
                            {f'<div class="route-line"></div>' if idx < len(rd['steps'])-1 else ''}
                        </div>
                        <div style="flex:1;">
                            <div style="font-size:0.9rem;color:#1e293b;">{s}</div>
                        </div>
                    </div>"""
                st.markdown(steps_html, unsafe_allow_html=True)

            if rd.get("lines"):
                st.markdown("**🚇 公交线路**")
                for l in rd["lines"]:
                    st.info(f"🚃 {l}")

            c1, c2 = st.columns(2)
            if c1.button("⭐ 收藏路线"):
                add_favorite({"type":"路线","name":f"{origin}→{dest}","detail":f"{mode} {rd.get('duration','')}"}, st.session_state.username)
                st.success("已收藏!")
            if c2.button("📍 加入行程"):
                add_itinerary_spot({"name":f"🗺️ {origin}→{dest}","city":"","note":f"{mode} {rd.get('duration','')}"}, st.session_state.username)
                st.success("已加入!")

    elif map_action == "周边搜索":
        loc = st.text_input("位置", "新宿站", key="nb_loc")
        kw = st.text_input("关键词", "餐厅", key="nb_kw")
        if st.button("🔍 搜索", type="primary", use_container_width=True):
            st.session_state.nearby_result = json.loads(AmapTool("nearby").run({"location_name":loc,"keywords":kw,"radius":3000}))
        nr = st.session_state.nearby_result
        if nr:
            for p in nr.get("places", []):
                # Visual place card
                st.markdown(f"""
                <div class="card">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <div style="font-weight:600;color:#1e293b;">{p['name']}</div>
                            <div style="font-size:0.8rem;color:#64748b;">📍{p.get('distance','')} | ⭐{p.get('rating','')} | 📞{p.get('tel','')}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                c1, c2 = st.columns([9, 1])
                c1.markdown("")
                if c2.button("⭐", key=f"fav_nb_{p.get('name','')}"):
                    add_favorite({"type":kw,"name":p["name"],"detail":f"{p.get('distance','')} ⭐{p.get('rating','')}"}, st.session_state.username)
                    st.success("已收藏!")

    elif map_action == "地理编码":
        addr = st.text_input("地址", "东京塔", key="geo_addr")
        if st.button("🔍 编码", use_container_width=True):
            st.json(json.loads(AmapTool("geocode").run({"address":addr})))

    elif map_action == "行政区划":
        kw = st.text_input("查询", "东京", key="dist_kw")
        if st.button("🔍 查询", use_container_width=True):
            st.json(json.loads(AmapTool("district").run({"keywords":kw})))

# ═══════════════ PAGE: 天气 ═══════════════
def page_weather():
    _back_button()
    _section_title("🌤️", "天气")
    w_dest = st.text_input("目的地", "东京", key="w_dest")
    img = _get_city_img(w_dest)
    if img:
        st.markdown(f'<img src="{img}" style="width:100%;max-height:150px;object-fit:cover;border-radius:12px;margin-bottom:1rem;">', unsafe_allow_html=True)
    if st.button("🌤️ 查询", type="primary", use_container_width=True):
        st.session_state.weather_data = json.loads(WeatherTool().run({"destination": w_dest, "days": 7}))
    wd = st.session_state.weather_data
    if wd:
        today = wd.get("today", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("天气", today.get("condition",""))
        c2.metric("温度", f"{today.get('temp_low','')}~{today.get('temp_high','')}")
        c3.metric("湿度", today.get("humidity",""))
        c4.metric("UV", f"{today.get('uv_index','')}({today.get('uv_label','')})")
        clothing = wd.get("clothing", {})
        if clothing:
            c1, c2, c3 = st.columns(3)
            c1.info(f"👕 {clothing.get('clothes','')}")
            c2.info(f"🎒 {clothing.get('extras','')}")
            c3.info(f"☂️ {clothing.get('umbrella','')}")
        for tip in wd.get("health_tips", []): st.warning(f"💊 {tip}")
        for day in wd.get("forecast", []):
            rain = int(day.get("rain_probability","0%").replace("%",""))
            icon = "🌧️" if rain > 50 else "🌦️" if rain > 20 else "☀️"
            with st.expander(f"{icon} {day['date']} {day['condition']} ({day['temp_low']}~{day['temp_high']})"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("高温", day.get("temp_high",""))
                c2.metric("低温", day.get("temp_low",""))
                c3.metric("降雨", day.get("rain_probability",""))
                c4.metric("UV", f"{day.get('uv_index','')}({day.get('uv_label','')})")
                if st.button(f"📌 添加天气备注", key=f"w_note_{day['date']}"):
                    save_note(f"weather_{day['date']}", f"{day['condition']} {day['temp_low']}~{day['temp_high']} 降雨{day['rain_probability']}", st.session_state.username)
                    st.success("已添加到行程备注!")
        if wd.get("suggestion"): st.info(wd["suggestion"])

# ═══════════════ PAGE: 预算 ═══════════════
def page_budget():
    _back_button()
    _section_title("💰", "预算管理")
    b_dest = st.text_input("目的地", "东京", key="b_dest")
    b_budget = st.number_input("总预算(CNY)", 1000, 200000, 15000, key="b_budget")
    b_days = st.number_input("天数", 1, 30, 5, key="b_days")
    b_style = st.selectbox("风格", ["舒适", "穷游", "豪华"], key="b_style")

    if st.button("💰 生成预算", type="primary", use_container_width=True):
        tool = BudgetTool()
        try: st.session_state.budget_data = json.loads(tool.run({"budget": b_budget, "days": b_days, "destination": b_dest, "style": b_style}))
        except: st.session_state.budget_data = {}

    bd = st.session_state.budget_data
    if bd:
        for alloc in bd.get("allocations", []):
            st.markdown(f"**{alloc.get('icon','💰')} {alloc['category']}** — ¥{alloc['amount']:,} ({alloc.get('percent',0)}%)")
            st.progress(alloc.get("percent", 0) / 100)
            if alloc.get("tip"): st.caption(f"💡 {alloc['tip']}")
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("每日(CNY)", f"¥{bd.get('daily_budget',0):,}")
        c2.metric(f"每日({bd.get('local_currency','')})", f"{bd.get('daily_local',0):,}")
        c3.metric("天数", f"{bd.get('days',b_days)}天")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        _section_title("📝", "实时记账")
        total = bd.get("total", b_budget)
        spent = sum(e["amount"] for e in st.session_state.expense_list)
        pct = spent / total * 100 if total else 0
        if pct > 90: st.error(f"⚠️ 超支预警！已花 {pct:.0f}%，剩 ¥{total-spent:,}")
        elif pct > 70: st.warning(f"已花 {pct:.0f}%，剩 ¥{total-spent:,}")
        else: st.success(f"预算健康 {pct:.0f}%，剩 ¥{total-spent:,}")
        st.progress(min(pct/100, 1.0))

        with st.form("add_exp_form"):
            ec1, ec2, ec3, ec4 = st.columns(4)
            exp_item = ec1.text_input("项目", "午餐")
            exp_amount = ec2.number_input("金额", 0, 100000, 100)
            exp_payer = ec3.text_input("付款人", "我")
            exp_cat = ec4.selectbox("分类", ["餐饮", "交通", "住宿", "门票", "购物", "其他"])
            if st.form_submit_button("➕ 添加"):
                st.session_state.expense_list.append({"item": exp_item, "amount": exp_amount, "payer": exp_payer, "category": exp_cat})

        if st.session_state.expense_list:
            cat_sum = {}
            for exp in st.session_state.expense_list:
                cat_sum[exp["category"]] = cat_sum.get(exp["category"], 0) + exp["amount"]
            st.markdown("**分类统计**")
            for cat, amt in sorted(cat_sum.items(), key=lambda x: -x[1]):
                st.markdown(f"{cat}: ¥{amt:,} ({amt/max(spent,1)*100:.0f}%)")
            st.markdown("---")
            for i, exp in enumerate(st.session_state.expense_list):
                c1, c2, c3, c4 = st.columns([4, 2, 2, 1])
                c1.markdown(f"• {exp['item']}")
                c2.caption(f"¥{exp['amount']:,}")
                c3.caption(f"{exp['category']}|{exp['payer']}")
                if c4.button("🗑️", key=f"del_exp_{i}"):
                    st.session_state.expense_list.pop(i)
                    st.rerun()
            if st.button("📊 导出报表"):
                os.makedirs("saved_reports", exist_ok=True)
                with open(f"saved_reports/{b_dest}_expense.md", "w", encoding="utf-8") as f:
                    f.write(f"# 费用报表\n\n总预算:¥{total:,} 已花:¥{spent:,}\n\n")
                    for cat, amt in sorted(cat_sum.items(), key=lambda x: -x[1]):
                        f.write(f"- {cat}: ¥{amt:,}\n")
                st.success("已导出!")

# ═══════════════ PAGE: 汇率 ═══════════════
def page_exchange():
    _back_button()
    _section_title("💱", "汇率换算 & 钱包")
    c1, c2 = st.columns(2)
    with c1:
        ex_amount = st.number_input("金额", 1.0, 1000000.0, 1000.0, key="ex_amt")
        ex_from = st.selectbox("从", ["CNY","JPY","USD","EUR","THB","GBP","KRW","AUD","SGD","HKD"])
    with c2:
        ex_to = st.selectbox("到", ["JPY","CNY","USD","EUR","THB","GBP","KRW","AUD","SGD","HKD"])
    if st.button("💱 换算", type="primary", use_container_width=True):
        r = json.loads(ExchangeTool().run({"amount":ex_amount,"from":ex_from,"to":ex_to}))
        st.metric(f"{ex_amount} {r.get('from_name',ex_from)}", f"{r['result']:.2f} {r.get('to_name',ex_to)}")
        st.caption(f"汇率: 1 {ex_from} = {r['rate']} {ex_to}")
        if r.get("tip"): st.info(r["tip"])

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    _section_title("💰", "我的钱包")
    wallet = get_wallet(st.session_state.username)
    et = ExchangeTool()
    total_cny = 0
    for curr, amt in wallet.items():
        if amt > 0:
            if curr == "CNY": total_cny += amt
            else:
                try:
                    r = json.loads(et.run({"amount":amt,"from":curr,"to":"CNY"}))
                    total_cny += r.get("result", 0)
                except: pass
    st.metric("钱包总计(≈CNY)", f"¥{total_cny:,.0f}")
    for curr in ["CNY","JPY","USD","EUR","THB","GBP","KRW"]:
        amt = wallet.get(curr, 0)
        new_amt = st.number_input(f"{curr}", 0.0, 1000000.0, float(amt), key=f"wl_{curr}")
        if new_amt != amt: set_wallet(curr, new_amt, st.session_state.username)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    _section_title("💡", "费用估算器")
    c1, c2 = st.columns(2)
    with c1:
        est_hotel = st.number_input("酒店/晚(CNY)", 0, 50000, 600, key="est_h")
        est_food = st.number_input("餐饮/天(CNY)", 0, 10000, 300, key="est_f")
    with c2:
        est_flight = st.number_input("机票(CNY)", 0, 50000, 3000, key="est_fl")
        est_days = st.number_input("天数", 1, 30, 5, key="est_d")
    est_total = est_flight + (est_hotel + est_food) * est_days
    st.metric("预估总费用", f"¥{est_total:,}", f"¥{est_total//est_days:,}/天")

# ═══════════════ PAGE: 翻译 ═══════════════
def page_translate():
    _back_button()
    _section_title("🌐", "翻译 & 短语本")
    lang = st.selectbox("语言", ["日语","英语","法语","泰语","韩语","意大利语","西班牙语","德语"], key="trans_lang")
    lang_codes = {"日语":"ja","英语":"en","法语":"fr","泰语":"th","韩语":"ko","意大利语":"it","西班牙语":"es","德语":"de"}
    custom = st.text_input("输入中文翻译", "", key="trans_custom", placeholder="如：谢谢，多少钱")
    scene = st.selectbox("或选场景", ["餐厅点菜","问路","购物","入住酒店","就医","紧急求助"], key="trans_scene")

    if st.button("🌐 翻译", type="primary", use_container_width=True):
        if custom:
            from tools.translate import TranslateTool
            r = json.loads(TranslateTool().run({"text":custom,"target":lang_codes.get(lang,"ja")}))
            st.metric("原文", custom)
            st.metric("翻译", r.get("translated",""))
            st.caption(f"🔊 {r.get('pronunciation','')}")
            if st.button("⭐ 收藏此短语"):
                save_phrase({"zh":custom,"foreign":r.get("translated",""),"pron":r.get("pronunciation",""),"lang":lang}, st.session_state.username)
                st.success("已收藏!")
        else:
            r = chat_json(TRANSLATE_SCENE_PROMPT.format(scene=scene,language=lang_codes.get(lang,"ja")),intent="translate_scene")
            if not r:
                try: r = json.loads(chat("",intent="translate_scene"))
                except: r = {}
            st.session_state.translate_data = r
    td = st.session_state.translate_data
    if td:
        for p in td.get("phrases",[]):
            c1, c2, c3, c4 = st.columns([2,3,2,1])
            c1.markdown(f"**🇨🇳 {p['zh']}**")
            c2.markdown(f"🌍 {p.get('foreign','')}")
            c3.caption(f"🔊 {p.get('pron','')}")
            if c4.button("⭐",key=f"sp_{p.get('zh','')}{p.get('foreign','')}"):
                save_phrase({"zh":p["zh"],"foreign":p.get("foreign",""),"pron":p.get("pron",""),"lang":lang}, st.session_state.username)
                st.success("已收藏!")

    saved = get_saved_phrases(st.session_state.username)
    if saved:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        _section_title("⭐", "我的短语本")
        for i, p in enumerate(saved):
            c1, c2, c3 = st.columns([3, 4, 1])
            c1.markdown(f"🇨🇳 {p.get('zh','')}")
            c2.markdown(f"🌍 {p.get('foreign','')} 🔊 {p.get('pron','')}")
            if c3.button("🗑️", key=f"del_ph_{i}"):
                remove_saved_phrase(i, st.session_state.username)
                st.rerun()

# ═══════════════ PAGE: 清单 ═══════════════
def page_packing():
    _back_button()
    _section_title("📋", "旅行清单")
    pk_dest = st.text_input("目的地", "东京", key="pk_dest")
    pk_season = st.selectbox("季节", ["春", "夏", "秋", "冬"], key="pk_season")
    pk_days = st.number_input("天数", 1, 30, 5, key="pk_days")
    pk_act = st.text_input("特殊活动", "温泉,摄影,购物", key="pk_act")

    if st.button("📋 生成清单", type="primary", use_container_width=True):
        result = chat_json(PACKING_PROMPT.format(destination=pk_dest, season=pk_season, days=pk_days, activities=pk_act), intent="packing")
        if not result:
            try: result = json.loads(chat("", intent="packing"))
            except: result = {}
        st.session_state.packing_list = result

    pl = st.session_state.packing_list
    if pl:
        checked_count = total_count = 0
        for cat in pl.get("categories", []):
            cat_checked = 0
            cat_total = len(cat.get("items", []))
            st.subheader(cat["name"])
            for item in cat.get("items", []):
                total_count += 1
                key = f"pk_{cat['name']}_{item['item']}"
                if key not in st.session_state: st.session_state[key] = item.get("checked", False)
                label = item["item"]
                p = item.get("priority", "")
                if p: label = f"{'🔴' if p=='必带' else '🟡' if p=='推荐' else '🟢'} {label}"
                if st.checkbox(label, value=st.session_state[key], key=key+"_cb"):
                    st.session_state[key] = True
                    checked_count += 1
                    cat_checked += 1
                else:
                    st.session_state[key] = False
                if item.get("tip"): st.caption(f"  💡 {item['tip']}")
            st.progress(cat_checked / max(cat_total, 1))
        st.divider()
        pct = checked_count / max(total_count, 1)
        st.metric("总进度", f"{checked_count}/{total_count}", f"{pct*100:.0f}%")
        st.progress(pct)
        c1, c2 = st.columns(2)
        if c1.button("✅ 全部标记已备"):
            for cat in pl.get("categories", []):
                for item in cat.get("items", []):
                    st.session_state[f"pk_{cat['name']}_{item['item']}"] = True
            st.rerun()
        if c2.button("🔄 重置"):
            for cat in pl.get("categories", []):
                for item in cat.get("items", []):
                    st.session_state[f"pk_{cat['name']}_{item['item']}"] = False
            st.rerun()

# ═══════════════ PAGE: 攻略 ═══════════════
def page_guide():
    _back_button()
    _section_title("📖", "攻略")
    g_dest = st.text_input("目的地", "东京", key="g_dest")
    img = _get_city_img(g_dest)
    if img:
        st.markdown(f'<img src="{img}" style="width:100%;max-height:180px;object-fit:cover;border-radius:12px;margin-bottom:1rem;">', unsafe_allow_html=True)
    if st.button("📖 生成攻略", type="primary", use_container_width=True):
        context = get_context(g_dest)
        result = chat(GUIDE_PROMPT.format(destination=g_dest, context=context), intent="guide")
        if not result or len(result) < 50: result = chat("", intent="guide")
        st.session_state.guide_text = result
    gt = st.session_state.guide_text
    if gt:
        st.markdown(gt)
        c1, c2 = st.columns(2)
        if c1.button("💾 保存"):
            os.makedirs("saved_guides", exist_ok=True)
            with open(f"saved_guides/{g_dest}_guide.md", "w", encoding="utf-8") as f: f.write(gt)
            st.success("已保存!")
        if c2.button("⭐ 收藏"):
            add_favorite({"type": "攻略", "name": f"{g_dest}攻略", "detail": gt[:80]}, st.session_state.username)
            st.success("已收藏!")

# ═══════════════ PAGE: SOS ═══════════════
def page_sos():
    _back_button()
    _section_title("🆘", "紧急求助")
    st.error("⚠️ 仅在真实紧急情况下使用！")
    sos_loc = st.text_input("位置", "东京新宿区", key="sos_loc")
    sos_country = st.text_input("国家", "日本", key="sos_country")
    if st.button("🆘 获取", type="primary", use_container_width=True):
        r = chat_json(SOS_PROMPT.format(location=sos_loc,country=sos_country),intent="sos")
        if not r:
            try: r = json.loads(chat("",intent="sos"))
            except: r = {}
        st.session_state.sos_data = r
    sos = st.session_state.sos_data
    if sos:
        em = sos.get("emergency",{})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🚔警察", em.get("police",""))
        c2.metric("🚑急救", em.get("ambulance_fire",""))
        c3.metric("🏛️使馆", em.get("china_embassy",""))
        c4.metric("📞热线", em.get("traveler_hotline",""))
        for nb in sos.get("nearby",[]):
            st.markdown(f"**{nb.get('type','')}**: {nb['name']} 📞{nb.get('phone','')} 📍{nb.get('distance','')}")
        for p in sos.get("phrases",[]):
            st.markdown(f"- 🇨🇳 **{p['zh']}** → {p.get('foreign','')} (🔊 {p.get('pron','')})")
            if st.button(f"⭐收藏", key=f"sos_fav_{p.get('zh','')}"):
                save_phrase({"zh":p["zh"],"foreign":p.get("foreign",""),"pron":p.get("pron",""),"lang":"紧急用语"}, st.session_state.username)
                st.success("已收藏!")
        if st.button("💾 保存紧急卡"):
            os.makedirs("saved_sos",exist_ok=True)
            card = f"# 🆘 紧急信息\n位置:{sos_loc} 国家:{sos_country}\n\n警察:{em.get('police','')}\n急救:{em.get('ambulance_fire','')}\n使馆:{em.get('china_embassy','')}\n"
            for p in sos.get("phrases",[]): card += f"- {p['zh']}→{p.get('foreign','')} ({p.get('pron','')})\n"
            with open(f"saved_sos/{sos_country}_emergency.md","w",encoding="utf-8") as f: f.write(card)
            st.success("已保存!")

# ═══════════════ PAGE: 对比 ═══════════════
def page_compare():
    _back_button()
    _section_title("📊", "方案对比")
    cmp_dest = st.text_input("目的地", "东京", key="cmp_dest")
    img = _get_city_img(cmp_dest)
    if img:
        st.markdown(f'<img src="{img}" style="width:100%;max-height:150px;object-fit:cover;border-radius:12px;margin-bottom:1rem;">', unsafe_allow_html=True)
    if st.button("📊 生成", type="primary", use_container_width=True):
        r = chat_json(COMPARE_PROMPT.format(destination=cmp_dest,days=5),intent="compare")
        if not r:
            try: r = json.loads(chat("",intent="compare"))
            except: r = {}
        st.session_state.compare_data = r
    cd = st.session_state.compare_data
    if cd:
        plans = cd.get("plans", [])
        if len(plans) >= 3:
            cols = st.columns(3)
            colors = ["#22c55e", "#3b82f6", "#a855f7"]
            labels = ["🌿 穷游版", "🏠 舒适版", "👑 豪华版"]
            for i, plan in enumerate(plans):
                with cols[i]:
                    st.markdown(f"""
                    <div class="card" style="border-top:3px solid {colors[i]};">
                        <div style="font-weight:700;font-size:1.1rem;">{labels[i]}</div>
                        <div style="font-size:1.5rem;font-weight:700;color:{colors[i]};margin:0.5rem 0;">¥{plan['budget']:,}</div>
                        <div style="font-size:0.8rem;color:#64748b;">日均 ¥{plan.get('daily_cost',0):,}</div>
                        <div style="font-size:0.85rem;margin-top:0.5rem;">{plan.get('style','')}</div>
                        <div style="font-size:0.8rem;color:#64748b;margin-top:0.3rem;">🏨 {plan.get('accommodation','')}</div>
                        <div style="font-size:0.8rem;color:#64748b;">🍽️ {plan.get('dining','')}</div>
                        <div style="font-size:0.8rem;color:#64748b;">🚃 {plan.get('transport','')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    for h in plan.get("highlights", []): st.markdown(f"- ✨ {h}")
                    if st.button(f"选择{plan['name']}", key=f"sel_{plan['name']}", use_container_width=True):
                        st.success(f"已选择「{plan['name']}」!")
        else:
            for plan in plans:
                with st.expander(f"{plan['name']} — ¥{plan['budget']:,}", expanded=True):
                    st.markdown(f"风格: {plan.get('style','')}")
                    st.markdown(f"住宿: {plan.get('accommodation','')} | 餐饮: {plan.get('dining','')} | 交通: {plan.get('transport','')}")
                    for h in plan.get("highlights", []): st.markdown(f"- ✨ {h}")

# ═══════════════ PAGE: AA分摊 ═══════════════
def page_settle():
    _back_button()
    _section_title("👥", "AA分摊")
    members_str = st.text_input("参与者", "小明,小红,小华", key="settle_m")
    members = [m.strip() for m in members_str.split(",")]
    with st.form("settle_form"):
        c1, c2, c3 = st.columns(3)
        s_payer = c1.selectbox("付款人", members)
        s_item = c2.text_input("项目", "晚餐")
        s_amount = c3.number_input("金额", 0, 100000, 500)
        if st.form_submit_button("➕ 添加"):
            st.session_state.setdefault("settle_expenses_list",[]).append({"payer":s_payer,"item":s_item,"amount":s_amount})
    sl = st.session_state.get("settle_expenses_list",[])
    if sl:
        for i, e in enumerate(sl):
            c1, c2, c3, c4 = st.columns([3,2,2,1])
            c1.markdown(f"• {e['item']}")
            c2.caption(f"¥{e['amount']:,}")
            c3.caption(e["payer"])
            if c4.button("🗑️",key=f"del_se_{i}"):
                sl.pop(i); st.rerun()
        total = sum(e["amount"] for e in sl)
        per = total / len(members)
        st.metric("总额", f"¥{total:,}")
        st.metric("人均", f"¥{per:,.0f}")
        balances = {m: {"paid":0,"share":per} for m in members}
        for e in sl: balances[e["payer"]]["paid"] += e["amount"]
        for m, b in balances.items():
            diff = b["paid"] - b["share"]
            if diff > 0: st.success(f"**{m}**: 已付¥{b['paid']:,}，应收¥{diff:,.0f}")
            elif diff < 0: st.error(f"**{m}**: 已付¥{b['paid']:,}，应付¥{abs(diff):,.0f}")
            else: st.info(f"**{m}**: 已付¥{b['paid']:,}，已结清✅")

# ═══════════════ PAGE: 签证 ═══════════════
def page_visa():
    _back_button()
    _section_title("🛂", "签证 & 材料追踪")
    v_dest = st.text_input("目的地", "日本", key="v_dest")
    if st.button("🛂 查询", type="primary", use_container_width=True):
        r = chat_json(VISA_PROMPT.format(destination=v_dest,departure="中国"),intent="visa")
        if not r:
            try: r = json.loads(chat("",intent="visa"))
            except: r = {}
        st.session_state.visa_data = r
    vd = st.session_state.visa_data
    if vd:
        c1, c2, c3 = st.columns(3)
        c1.metric("需要签证", "✅是" if vd.get("visa_required") else "🛂免签", vd.get("visa_type",""))
        c2.metric("停留", vd.get("stay_duration",""))
        c3.metric("办理时间", vd.get("processing_time",""))
        reqs = vd.get("requirements",[])
        if reqs:
            st.subheader("📋 材料清单")
            checks = get_visa_check(v_dest, st.session_state.username)
            done = 0
            for req in reqs:
                checked = checks.get(req, False)
                if st.checkbox(req, value=checked, key=f"visa_{v_dest}_{req[:20]}"):
                    set_visa_check(v_dest, req, True, st.session_state.username); done += 1
                else:
                    set_visa_check(v_dest, req, False, st.session_state.username)
            progress = done / len(reqs) if reqs else 0
            st.progress(progress)
            st.metric("进度", f"{done}/{len(reqs)}", f"{progress*100:.0f}%")
            if progress == 1.0: st.success("🎉 所有材料已备齐！")
            elif progress >= 0.5: st.info("过半了，继续加油！")
            if st.button("🔔 提醒我办签证"):
                add_reminder({"text": f"办理{v_dest}签证","date": "尽快","type": "签证"}, st.session_state.username)
                st.success("已添加提醒!")
        for tip in vd.get("tips",[]): st.info(tip)
        if vd.get("embassy_info"): st.info(f"🏛️ {vd['embassy_info']}")

# ═══════════════ PAGE: 日记 ═══════════════
def page_diary():
    _back_button()
    _section_title("📝", "旅行日记")
    trip = st.session_state.current_trip
    if trip:
        if st.button("📝 生成", type="primary", use_container_width=True):
            r = chat(DIARY_PROMPT.format(trip_data=json.dumps(trip,ensure_ascii=False)),intent="diary")
            if not r or len(r)<50: r = chat("",intent="diary")
            st.session_state.diary_text = r
        dt = st.session_state.diary_text
        if dt:
            st.markdown(dt)
            c1, c2 = st.columns(2)
            if c1.button("💾 保存"):
                os.makedirs("saved_diaries",exist_ok=True)
                with open("saved_diaries/latest.md","w",encoding="utf-8") as f: f.write(dt)
                st.success("已保存!")
            if c2.button("⭐ 收藏"):
                add_favorite({"type":"日记","name":"旅行日记","detail":dt[:80]}, st.session_state.username)
                st.success("已收藏!")
    else:
        st.info("请先规划行程")

# ═══════════════ PAGE: 知识库 ═══════════════
def page_knowledge():
    _back_button()
    _section_title("🔍", "知识库")
    q = st.text_input("搜索", "东京 美食", key="rag_q")
    if st.button("🔍 搜索", type="primary", use_container_width=True):
        st.session_state.search_results = search(q, 8)
    results = st.session_state.search_results
    if results:
        for r in results:
            sec = r.get("section","")
            with st.expander(f"📍 {r['city']} [{sec}] (相关度:{r.get('score',0):.3f})"):
                st.markdown(r["content"])
                if st.button("⭐ 收藏", key=f"fav_rag_{r['city']}_{r.get('section','')}_{r.get('score',0)}"):
                    add_favorite({"type":"知识","name":f"{r['city']}[{sec}]","detail":r["content"][:100]}, st.session_state.username)
                    st.success("已收藏!")

# ═══════════════ PAGE: 提醒 ═══════════════
def page_reminders():
    _back_button()
    _section_title("🔔", "提醒 & 待办")
    with st.form("add_reminder_form"):
        c1, c2, c3 = st.columns(3)
        rem_text = c1.text_input("内容", "", placeholder="如：7月15号办签证")
        rem_date = c2.text_input("日期", "", placeholder="如：2025-07-15")
        rem_type = c3.selectbox("类型", ["旅行提醒", "签证", "预订确认", "打包准备", "其他"])
        if st.form_submit_button("➕ 添加提醒"):
            if rem_text:
                add_reminder({"text": rem_text, "date": rem_date, "type": rem_type}, st.session_state.username)
                st.success("已添加!")
                st.rerun()

    reminders = get_reminders(st.session_state.username)
    if reminders:
        pending = [r for r in reminders if not r.get("done")]
        done_list = [r for r in reminders if r.get("done")]
        if pending:
            st.subheader("📋 待办")
            for r in pending:
                c1, c2, c3 = st.columns([8, 1, 1])
                icon = {"旅行提醒":"📌","签证":"🛂","预订确认":"🎫","打包准备":"📋"}.get(r.get("type",""),"🔔")
                c1.markdown(f"{icon} **{r.get('text','')}** — {r.get('date','')} ({r.get('type','')})")
                if c2.button("✅", key=f"done_{r['id']}"):
                    toggle_reminder(r["id"], st.session_state.username); st.rerun()
                if c3.button("🗑️", key=f"del_{r['id']}"):
                    remove_reminder(r["id"], st.session_state.username); st.rerun()
        if done_list:
            with st.expander(f"✅ 已完成 ({len(done_list)})"):
                for r in done_list:
                    c1, c2 = st.columns([9, 1])
                    c1.caption(f"~~{r.get('text','')}~~ ({r.get('date','')})")
                    if c2.button("🗑️", key=f"del_done_{r['id']}"):
                        remove_reminder(r["id"], st.session_state.username); st.rerun()
    else:
        st.info("暂无提醒。在 AI 助手中说「提醒我XX」可快速添加！")

# ═══════════════ PAGE: 收藏 ═══════════════
def page_favorites():
    _back_button()
    _section_title("⭐", "收藏夹")
    favorites = get_favorites(username=st.session_state.username)
    if favorites:
        fav_by_type = {}
        for f in favorites:
            t = f.get("type","其他")
            fav_by_type.setdefault(t,[]).append(f)
        for ftype, items in fav_by_type.items():
            icon = {"景点":"📍","酒店":"🏨","航班":"✈️","路线":"🗺️","攻略":"📖","知识":"📚","翻译":"🌐","日记":"📝","行程":"🗺️"}.get(ftype,"⭐")
            st.subheader(f"{icon} {ftype} ({len(items)})")
            for item in items:
                c1, c2, c3 = st.columns([5, 3, 1])
                c1.markdown(f"**{item.get('name','')}**")
                c2.caption(item.get("detail",""))
                if c3.button("🗑️", key=f"rm_fav_{item['id']}"):
                    remove_favorite(item["id"], st.session_state.username); st.rerun()
        if st.button("🗑️ 清空所有收藏"):
            data = _load()
            if st.session_state.username in data.get("users", {}):
                data["users"][st.session_state.username]["favorites"] = []
                _save(data)
            st.success("已清空!")
    else:
        st.info("暂无收藏。在各模块中点击⭐收藏感兴趣的内容。")

# ═══════════════ PAGE: 设置 ═══════════════
def page_settings():
    _back_button()
    _section_title("⚙️", "设置")
    st.subheader("🤖 AI 配置")
    if is_llm_available():
        st.success("LLM API 已连接")
    else:
        st.info("当前为 Mock 模式。在 .env 文件中配置 LLM_API_KEY 可启用真实 AI")
        st.code("LLM_API_KEY=sk-xxx\nLLM_BASE_URL=https://api.openai.com/v1\nLLM_MODEL=gpt-4o-mini")

    if os.getenv("AMAP_API_KEY"):
        st.success("高德地图 API 已连接")
    else:
        st.info("当前为高德 Mock 模式。在 .env 中配置 AMAP_API_KEY 可启用真实 API")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.subheader("🗑️ 数据管理")
    c1, c2, c3 = st.columns(3)
    if c1.button("清除对话历史"):
        clear_session(st.session_state.session_id)
        st.session_state.chat_history = []
        st.success("已清除!")
    if c2.button("重置偏好"):
        update_profile({"budget_range":"","travel_style":"","preferred_season":"","accommodation":"","dietary":"","interests":"","avoid":""}, st.session_state.username)
        st.success("已重置!")
    if c3.button("清除所有数据"):
        data = _load()
        if st.session_state.username in data.get("users", {}):
            data["users"][st.session_state.username] = {
                "password": data["users"][st.session_state.username]["password"],
                "nickname": data["users"][st.session_state.username]["nickname"],
                "avatar": data["users"][st.session_state.username].get("avatar", "🧑‍✈️"),
                "created_at": data["users"][st.session_state.username].get("created_at", ""),
                "profile": {}, "bookings": [], "favorites": [], "itinerary_spots": [],
                "saved_phrases": [], "visa_checklist": {}, "notes": {},
                "trip_log": [], "currency_wallet": {}, "reminders": [],
            }
            _save(data)
        st.success("数据已清除!")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.subheader("ℹ️ 关于")
    st.markdown("""
    **TravelMate v3.0** — AI智能出行助手

    功能：行程规划 | 航班酒店预订 | 天气查询 | 汇率换算 | 翻译短语本 |
    签证材料追踪 | AA分摊 | 旅行日记 | SOS求助 | 知识库搜索 |
    个人主页 | 提醒待办 | 收藏夹 | 费用记账

    技术栈：FastAPI + Streamlit + ReAct Agent + TF-IDF/BM25 RAG + 高德地图
    """)

# ═══════════════ ROUTER ═══════════════
PAGE_MAP = {
    "主页": page_home,
    "个人主页": page_profile,
    "AI助手": page_chat,
    "行程规划": page_plan,
    "预算": page_budget,
    "清单": page_packing,
    "攻略": page_guide,
    "航班": page_flight,
    "酒店": page_hotel,
    "景点": page_attraction,
    "地图": page_map,
    "天气": page_weather,
    "汇率": page_exchange,
    "翻译": page_translate,
    "SOS": page_sos,
    "对比": page_compare,
    "AA分摊": page_settle,
    "签证": page_visa,
    "日记": page_diary,
    "知识库": page_knowledge,
    "提醒": page_reminders,
    "收藏": page_favorites,
    "设置": page_settings,
}

if not st.session_state.logged_in:
    page_auth()
else:
    render_sidebar()
    current_page = st.session_state.current_page
    page_func = PAGE_MAP.get(current_page, page_home)
    page_func()
