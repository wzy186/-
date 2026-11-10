"""SQLite 持久化：快照与方案两张表。

设计要点（详见 DESIGN.md）：
- 启动时把种子 JSON 整体写入 snapshots 表（INSERT OR IGNORE），
  方案重开时始终以库内快照为准，历史方案不受种子文件后续变更影响；
- plans 表只保存操作列表（ops_json）与快照关联，分析结果读取时重算，
  操作列表是唯一事实来源；
- 原始组合从不写入数据库，隔离性由构造保证。
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id    TEXT PRIMARY KEY,
    portfolio_id   TEXT NOT NULL,
    portfolio_name TEXT NOT NULL,
    currency       TEXT NOT NULL,
    snapshot_at    TEXT NOT NULL,
    payload        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    ops_json    TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def init_db(db_path: Path, seed_payload: dict) -> None:
    """建表并把种子快照写入库（幂等）。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute(
            """
            INSERT OR IGNORE INTO snapshots (snapshot_id, portfolio_id, portfolio_name, currency, snapshot_at, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                seed_payload["snapshot_id"],
                seed_payload["portfolio_id"],
                seed_payload["name"],
                seed_payload["currency"],
                seed_payload["snapshot_at"],
                json.dumps(seed_payload, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def connect(db_path: Path) -> sqlite3.Connection:
    """单用户场景：每个请求一个短连接，无需连接池。"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def insert_plan(db_path: Path, name: str, snapshot_id: str, ops_json: str, created_at: str) -> int:
    conn = connect(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO plans (name, snapshot_id, ops_json, created_at) VALUES (?, ?, ?, ?)",
            (name, snapshot_id, ops_json, created_at),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def list_plans(db_path: Path) -> list[dict]:
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, name, snapshot_id, created_at FROM plans ORDER BY created_at DESC, id DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_plan(db_path: Path, plan_id: int) -> dict[str, Any] | None:
    conn = connect(db_path)
    try:
        row = conn.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_plan(db_path: Path, plan_id: int) -> None:
    conn = connect(db_path)
    try:
        conn.execute("DELETE FROM plans WHERE id = ?", (plan_id,))
        conn.commit()
    finally:
        conn.close()


def get_snapshot_payload(db_path: Path, snapshot_id: str) -> dict | None:
    conn = connect(db_path)
    try:
        row = conn.execute("SELECT payload FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
        return json.loads(row["payload"]) if row else None
    finally:
        conn.close()
