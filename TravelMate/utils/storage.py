"""Persistent storage for bookings, favorites, itinerary, phrases, visa checklist, notes, user profile.
Supports multi-user: all data is scoped per user_id."""
import os
import json
import uuid
import hashlib
from datetime import datetime

_STORAGE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".travel_data.json")

def _load():
    if os.path.exists(_STORAGE_FILE):
        try:
            with open(_STORAGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"users": {}, "current_user": None}

def _save(data):
    with open(_STORAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()

# ── Auth ──
def register_user(username: str, password: str, nickname: str = "") -> dict:
    data = _load()
    users = data.setdefault("users", {})
    if username in users:
        return {"success": False, "error": "用户名已存在"}
    users[username] = {
        "password": _hash_password(password),
        "nickname": nickname or username,
        "avatar": "🧑‍✈️",
        "created_at": datetime.now().isoformat(),
        "profile": {},
        "bookings": [], "favorites": [], "itinerary_spots": [],
        "saved_phrases": [], "visa_checklist": {}, "notes": {},
        "trip_log": [], "currency_wallet": {}, "reminders": [],
    }
    _save(data)
    return {"success": True, "username": username}

def login_user(username: str, password: str) -> dict:
    data = _load()
    users = data.get("users", {})
    if username not in users:
        return {"success": False, "error": "用户名不存在"}
    if users[username]["password"] != _hash_password(password):
        return {"success": False, "error": "密码错误"}
    data["current_user"] = username
    _save(data)
    return {"success": True, "username": username, "nickname": users[username].get("nickname", username)}

def logout_user():
    data = _load()
    data["current_user"] = None
    _save(data)

def get_current_user() -> str | None:
    return _load().get("current_user")

def get_user_data(username: str) -> dict:
    data = _load()
    return data.get("users", {}).get(username, {})

def set_current_user(username: str):
    data = _load()
    data["current_user"] = username
    _save(data)

# ── Helpers to get user-scoped data ──
def _user_load(username: str) -> dict:
    data = _load()
    return data.get("users", {}).get(username, {})

def _user_save(username: str, user_data: dict):
    data = _load()
    data.setdefault("users", {})[username] = user_data
    _save(data)

# ── Profile ──
def get_profile(username: str = None) -> dict:
    u = username or get_current_user()
    if not u:
        return {}
    return _user_load(u).get("profile", {})

def update_profile(profile: dict, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    ud.setdefault("profile", {}).update(profile)
    _user_save(u, ud)

def get_nickname(username: str = None) -> str:
    u = username or get_current_user()
    if not u:
        return "旅行者"
    return _user_load(u).get("nickname", u)

def set_nickname(nickname: str, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    ud["nickname"] = nickname
    _user_save(u, ud)

def get_avatar(username: str = None) -> str:
    u = username or get_current_user()
    if not u:
        return "🧑‍✈️"
    return _user_load(u).get("avatar", "🧑‍✈️")

def set_avatar(avatar: str, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    ud["avatar"] = avatar
    _user_save(u, ud)

# ── Bookings ──
def add_booking(item, username: str = None):
    u = username or get_current_user()
    if not u:
        return None
    ud = _user_load(u)
    item["id"] = str(uuid.uuid4())[:8]
    item["booked_at"] = datetime.now().isoformat()
    item["status"] = item.get("status", "已确认")
    ud.setdefault("bookings", []).append(item)
    _user_save(u, ud)
    return item["id"]

def update_booking(booking_id, updates, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    for b in ud.get("bookings", []):
        if b["id"] == booking_id:
            b.update(updates)
            break
    _user_save(u, ud)

def remove_booking(booking_id, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    ud["bookings"] = [b for b in ud.get("bookings", []) if b["id"] != booking_id]
    _user_save(u, ud)

def get_bookings(booking_type=None, username: str = None):
    u = username or get_current_user()
    if not u:
        return []
    bookings = _user_load(u).get("bookings", [])
    if booking_type:
        return [b for b in bookings if b.get("type") == booking_type]
    return bookings

# ── Favorites ──
def add_favorite(item, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    for f in ud.get("favorites", []):
        if f.get("name") == item.get("name") and f.get("type") == item.get("type"):
            return
    item["id"] = str(uuid.uuid4())[:8]
    item["saved_at"] = datetime.now().isoformat()
    ud.setdefault("favorites", []).append(item)
    _user_save(u, ud)

def remove_favorite(fav_id, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    ud["favorites"] = [f for f in ud.get("favorites", []) if f["id"] != fav_id]
    _user_save(u, ud)

def get_favorites(fav_type=None, username: str = None):
    u = username or get_current_user()
    if not u:
        return []
    favs = _user_load(u).get("favorites", [])
    return [f for f in favs if f.get("type") == fav_type] if fav_type else favs

# ── Itinerary ──
def add_itinerary_spot(spot, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    ud.setdefault("itinerary_spots", []).append(spot)
    _user_save(u, ud)

def remove_itinerary_spot(index, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    spots = ud.get("itinerary_spots", [])
    if 0 <= index < len(spots):
        spots.pop(index)
    _user_save(u, ud)

def get_itinerary_spots(username: str = None):
    u = username or get_current_user()
    if not u:
        return []
    return _user_load(u).get("itinerary_spots", [])

def clear_itinerary_spots(username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    ud["itinerary_spots"] = []
    _user_save(u, ud)

# ── Phrases ──
def save_phrase(phrase, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    ud.setdefault("saved_phrases", []).append(phrase)
    _user_save(u, ud)

def get_saved_phrases(username: str = None):
    u = username or get_current_user()
    if not u:
        return []
    return _user_load(u).get("saved_phrases", [])

def remove_saved_phrase(index, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    phrases = ud.get("saved_phrases", [])
    if 0 <= index < len(phrases):
        phrases.pop(index)
    _user_save(u, ud)

# ── Visa ──
def set_visa_check(destination, item, checked, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    ud.setdefault("visa_checklist", {}).setdefault(destination, {})[item] = checked
    _user_save(u, ud)

def get_visa_check(destination, username: str = None):
    u = username or get_current_user()
    if not u:
        return {}
    return _user_load(u).get("visa_checklist", {}).get(destination, {})

# ── Notes ──
def save_note(key, content, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    ud.setdefault("notes", {})[key] = content
    _user_save(u, ud)

def get_note(key, username: str = None):
    u = username or get_current_user()
    if not u:
        return ""
    return _user_load(u).get("notes", {}).get(key, "")

def get_all_notes(username: str = None):
    u = username or get_current_user()
    if not u:
        return {}
    return _user_load(u).get("notes", {})

# ── Trip log ──
def add_trip_log(entry, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    entry["id"] = str(uuid.uuid4())[:8]
    entry["created_at"] = datetime.now().isoformat()
    ud.setdefault("trip_log", []).append(entry)
    _user_save(u, ud)

def get_trip_log(username: str = None):
    u = username or get_current_user()
    if not u:
        return []
    return _user_load(u).get("trip_log", [])

def update_trip_log(trip_id, updates, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    for t in ud.get("trip_log", []):
        if t.get("id") == trip_id:
            t.update(updates)
            break
    _user_save(u, ud)

# ── Currency wallet ──
def set_wallet(currency, amount, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    ud.setdefault("currency_wallet", {})[currency] = amount
    _user_save(u, ud)

def get_wallet(username: str = None):
    u = username or get_current_user()
    if not u:
        return {}
    return _user_load(u).get("currency_wallet", {})

# ── Reminders ──
def add_reminder(reminder, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    reminder["id"] = str(uuid.uuid4())[:8]
    reminder["created_at"] = datetime.now().isoformat()
    reminder["done"] = False
    ud.setdefault("reminders", []).append(reminder)
    _user_save(u, ud)

def get_reminders(username: str = None):
    u = username or get_current_user()
    if not u:
        return []
    return _user_load(u).get("reminders", [])

def toggle_reminder(reminder_id, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    for r in ud.get("reminders", []):
        if r["id"] == reminder_id:
            r["done"] = not r["done"]
            break
    _user_save(u, ud)

def remove_reminder(reminder_id, username: str = None):
    u = username or get_current_user()
    if not u:
        return
    ud = _user_load(u)
    ud["reminders"] = [r for r in ud.get("reminders", []) if r["id"] != reminder_id]
    _user_save(u, ud)

# ── Stats ──
def get_stats(username: str = None):
    u = username or get_current_user()
    if not u:
        return {"total_trips": 0, "total_bookings": 0, "total_favorites": 0, "total_phrases": 0, "total_reminders": 0}
    ud = _user_load(u)
    return {
        "total_trips": len(ud.get("trip_log", [])),
        "total_bookings": len(ud.get("bookings", [])),
        "total_favorites": len(ud.get("favorites", [])),
        "total_phrases": len(ud.get("saved_phrases", [])),
        "total_reminders": len([r for r in ud.get("reminders", []) if not r.get("done")]),
    }
