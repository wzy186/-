import json
import os
from datetime import datetime

MEMORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".user_memory.json")
MAX_SESSION_MSGS = 50
MAX_SESSIONS = 20


def _load() -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {"profiles": {}, "sessions": {}, "trip_history": {}, "preferences": {}}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("profiles", {})
        data.setdefault("sessions", {})
        data.setdefault("trip_history", {})
        data.setdefault("preferences", {})
        return data
    except Exception:
        return {"profiles": {}, "sessions": {}, "trip_history": {}, "preferences": {}}


def _save(data: dict):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _uid() -> str:
    """Get current user ID from storage's current_user field."""
    try:
        from utils.storage import get_current_user
        return get_current_user() or "default"
    except Exception:
        return "default"


def get_profile() -> dict:
    uid = _uid()
    return _load().get("profiles", {}).get(uid, {})


def update_profile(profile: dict):
    uid = _uid()
    data = _load()
    data.setdefault("profiles", {}).setdefault(uid, {}).update(profile)
    _save(data)


def add_session_message(session_id: str, role: str, content: str):
    data = _load()
    uid = _uid()
    user_sessions = data.setdefault("sessions", {}).setdefault(uid, {})
    msgs = user_sessions.setdefault(session_id, [])
    msgs.append({"role": role, "content": content, "ts": datetime.now().isoformat()})
    if len(msgs) > MAX_SESSION_MSGS:
        msgs[:] = msgs[-MAX_SESSION_MSGS:]
    if len(user_sessions) > MAX_SESSIONS:
        keys = list(user_sessions.keys())
        for k in keys[:-MAX_SESSIONS]:
            del user_sessions[k]
    _save(data)


def get_session(session_id: str) -> list[dict]:
    uid = _uid()
    return _load().get("sessions", {}).get(uid, {}).get(session_id, [])


def get_recent_sessions(n: int = 5) -> list[dict]:
    uid = _uid()
    data = _load()
    sessions = data.get("sessions", {}).get(uid, {})
    keys = list(sessions.keys())[-n:]
    result = []
    for k in keys:
        msgs = sessions[k]
        user_msgs = [m for m in msgs if m["role"] == "user"]
        asst_msgs = [m for m in msgs if m["role"] == "assistant"]
        last_user = user_msgs[-1]["content"][:100] if user_msgs else ""
        last_asst = asst_msgs[-1]["content"][:100] if asst_msgs else ""
        result.append({
            "session_id": k,
            "last_user_msg": last_user,
            "last_assistant_msg": last_asst,
            "message_count": len(msgs),
            "last_ts": msgs[-1].get("ts", "") if msgs else "",
        })
    return result


def add_trip_history(trip: dict):
    uid = _uid()
    data = _load()
    history = data.setdefault("trip_history", {}).setdefault(uid, [])
    trip["saved_at"] = datetime.now().isoformat()
    history.append(trip)
    if len(history) > 30:
        history[:] = history[-30:]
    _save(data)


def get_trip_history() -> list[dict]:
    uid = _uid()
    return _load().get("trip_history", {}).get(uid, [])


def set_preference(key: str, value):
    uid = _uid()
    data = _load()
    data.setdefault("preferences", {}).setdefault(uid, {})[key] = value
    _save(data)


def get_preference(key: str, default=None):
    uid = _uid()
    return _load().get("preferences", {}).get(uid, {}).get(key, default)


def get_all_preferences() -> dict:
    uid = _uid()
    return _load().get("preferences", {}).get(uid, {})


def clear_session(session_id: str):
    uid = _uid()
    data = _load()
    data.get("sessions", {}).get(uid, {}).pop(session_id, None)
    _save(data)
