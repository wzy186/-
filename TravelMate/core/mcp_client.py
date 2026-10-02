"""通用 MCP (Model Context Protocol) 客户端 — 接入外部 MCP Server。

支持两种传输协议:
- stdio: 子进程 + 换行分隔 JSON-RPC 2.0（如 npx/uvx 启动的本地 server，如 12306-mcp）
- http:  Streamable HTTP（POST JSON-RPC，响应为 JSON 或 SSE，如高德官方 MCP）

配置方式（环境变量 / .env），按 <NAME> 分组：
  MCP_<NAME>_MODE=http|stdio
  MCP_<NAME>_URL=https://mcp.amap.com/mcp?key=YOUR_KEY        # http 模式
  MCP_<NAME>_COMMAND=npx -y 12306-mcp                          # stdio 模式
  MCP_<NAME>_ENV={"KEY":"value"}                               # 可选，子进程环境变量
  MCP_<NAME>_INIT_TIMEOUT=60                                   # 可选，初始化超时秒数

示例：
  MCP_AMAP_MODE=http
  MCP_AMAP_URL=https://mcp.amap.com/mcp?key=xxx

  MCP_12306_MODE=stdio
  MCP_12306_COMMAND=npx -y 12306-mcp

远端工具注册到统一工具注册表，命名为 mcp_<name>_<tool>。
连接失败 / 未配置时静默降级，不影响本地 Mock 工具。
"""

from __future__ import annotations

import json
import os
import re
import select
import shlex
import subprocess
import threading
import time
from typing import Any

import httpx

from core.mcp import ToolSchema, register_tool

_PROTOCOL_VERSION = "2024-11-05"
_DEFAULT_CALL_TIMEOUT = 25.0

# 服务名 -> {"status": ..., "tools": [...], "error": ...}
MCP_SERVER_STATUS: dict[str, dict] = {}
_status_lock = threading.Lock()
_registered = threading.Event()


# ────────────────────────── JSON-RPC 连接 ──────────────────────────


class McpConnection:
    """单个 MCP Server 连接（stdio 或 streamable http）。"""

    def __init__(self, name: str, mode: str, url: str = "", command: str = "",
                 env: dict | None = None, init_timeout: float = 60.0):
        self.name = name
        self.mode = mode  # "stdio" | "http"
        self.url = url
        self.command = command
        self.env_extra = env or {}
        self.init_timeout = init_timeout
        self._id = 0
        # stdio
        self._proc: subprocess.Popen | None = None
        self._buf = b""
        # http
        self._session_id = ""
        self.tools: list[dict] = []

    # ── JSON-RPC core ──

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _rpc(self, method: str, params: dict | None = None,
             timeout: float = _DEFAULT_CALL_TIMEOUT, notify: bool = False) -> dict | None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        if notify:
            if self.mode == "stdio":
                self._send_line(json.dumps(payload, ensure_ascii=False))
            else:
                self._post(payload, timeout)
            return None
        rid = self._next_id()
        payload["id"] = rid
        if self.mode == "stdio":
            self._send_line(json.dumps(payload, ensure_ascii=False))
            deadline = time.time() + timeout
            while True:
                msg = self._read_msg(deadline)
                if msg.get("id") == rid:
                    if "error" in msg:
                        raise RuntimeError(f"MCP {self.name} {method} error: {msg['error']}")
                    return msg.get("result") or {}
                # 忽略通知/其他响应
        else:
            resp = self._post(payload, timeout)
            if "error" in resp:
                raise RuntimeError(f"MCP {self.name} {method} error: {resp['error']}")
            return resp.get("result") or {}

    # ── stdio transport ──

    def _ensure_proc(self):
        if self._proc and self._proc.poll() is None:
            return
        argv = shlex.split(self.command)
        if not argv:
            raise RuntimeError(f"MCP {self.name}: 空命令")
        env = {**os.environ, **self.env_extra}
        self._proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, bufsize=0,
        )
        self._buf = b""

    def _send_line(self, text: str):
        self._ensure_proc()
        self._proc.stdin.write((text + "\n").encode("utf-8"))
        self._proc.stdin.flush()

    def _read_msg(self, deadline: float) -> dict:
        while True:
            line = self._readline(deadline).strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue  # 跳过非 JSON 行

    def _readline(self, deadline: float) -> str:
        while b"\n" not in self._buf:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"MCP {self.name} stdio 读取超时")
            r, _, _ = select.select([self._proc.stdout], [], [], remaining)
            if not r:
                raise TimeoutError(f"MCP {self.name} stdio 读取超时")
            chunk = os.read(self._proc.stdout.fileno(), 65536)
            if not chunk:
                raise ConnectionError(f"MCP {self.name} 已关闭 stdout")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("utf-8", "replace")

    # ── streamable http transport ──

    def _post(self, payload: dict, timeout: float) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        r = httpx.post(self.url, json=payload, headers=headers, timeout=timeout)
        sid = r.headers.get("Mcp-Session-Id") or r.headers.get("mcp-session-id")
        if sid:
            self._session_id = sid
        r.raise_for_status()
        body = r.text.strip()
        if not body:
            return {}
        if body.startswith("{") or body.startswith("["):
            return json.loads(body)
        # SSE: 解析 data: 行
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if not data:
                    continue
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") is not None or "result" in msg or "error" in msg:
                    return msg
        return {}

    # ── MCP lifecycle ──

    def connect(self):
        result = self._rpc("initialize", {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "TravelMate", "version": "2.0"},
        }, timeout=self.init_timeout)
        # 兼容 server 返回更高协议版本
        server_ver = (result or {}).get("protocolVersion")
        if server_ver:
            globals()["_PROTOCOL_VERSION"] = server_ver
        self._rpc("notifications/initialized", {}, notify=True)

    def list_tools(self) -> list[dict]:
        result = self._rpc("tools/list", {}) or {}
        self.tools = result.get("tools", [])
        return self.tools

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments},
                           timeout=_DEFAULT_CALL_TIMEOUT) or {}
        if result.get("isError"):
            texts = _extract_texts(result)
            raise RuntimeError(texts or f"工具 {name} 执行失败")
        texts = _extract_texts(result)
        return "\n".join(texts) if texts else json.dumps(result, ensure_ascii=False)

    def close(self):
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None


def _extract_texts(result: dict) -> list[str]:
    texts = []
    for item in result.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            texts.append(item.get("text", ""))
    return texts


# ────────────────────────── 配置发现与注册 ──────────────────────────


def discover_servers() -> list[dict]:
    """从环境变量解析 MCP server 配置。"""
    groups: dict[str, dict] = {}
    pattern = re.compile(r"^MCP_([A-Za-z0-9]+)_(MODE|URL|COMMAND|ENV|INIT_TIMEOUT)$")
    for key, val in os.environ.items():
        m = pattern.match(key)
        if not m or not val:
            continue
        name, field = m.group(1).lower(), m.group(2).lower()
        cfg = groups.setdefault(name, {"name": name})
        if field == "mode":
            cfg["mode"] = val.strip().lower()
        elif field == "url":
            cfg["url"] = val.strip()
        elif field == "command":
            cfg["command"] = val.strip()
        elif field == "env":
            try:
                cfg["env"] = json.loads(val)
            except json.JSONDecodeError:
                pass
        elif field == "init_timeout":
            try:
                cfg["init_timeout"] = float(val)
            except ValueError:
                pass
    return [c for c in groups.values() if c.get("mode") in ("http", "stdio")]


_ACTION_HINTS = ("book", "order", "reserve", "submit", "pay", "create", "cancel", "buy", "订", "预订", "下单", "购买")


def _guess_category(tool_name: str, description: str) -> str:
    text = f"{tool_name} {description}".lower()
    return "action" if any(k in text for k in _ACTION_HINTS) else "query"


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name.lower()).strip("_")


def _register_one_server(cfg: dict):
    name = cfg["name"]
    conn = McpConnection(
        name=name, mode=cfg["mode"], url=cfg.get("url", ""),
        command=cfg.get("command", ""), env=cfg.get("env"),
        init_timeout=cfg.get("init_timeout", 60.0),
    )
    conn.connect()
    tools = conn.list_tools()
    registered = []
    for t in tools:
        tool_name = t.get("name", "")
        if not tool_name:
            continue
        reg_name = f"mcp_{_sanitize(name)}_{_sanitize(tool_name)}"
        params = t.get("inputSchema") or {"type": "object", "properties": {}}
        if not isinstance(params, dict) or "type" not in params:
            params = {"type": "object", "properties": params or {}}
        desc = f"[{name} MCP] {t.get('description', '')}".strip()
        category = _guess_category(tool_name, t.get("description", ""))

        def _executor(_n: str, args: dict, _conn=conn, _tn=tool_name, _is_action=(category == "action")):
            try:
                text = _conn.call_tool(_tn, args)
            except Exception as e:
                return json.dumps({"success": False, "error": f"MCP 调用失败: {e}",
                                   "message": f"⚠️ MCP 工具 {_tn} 调用失败: {e}"}, ensure_ascii=False), _is_action
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {"raw": text}
            if isinstance(data, dict) and "message" not in data:
                data["message"] = str(text)[:300]
            return json.dumps(data, ensure_ascii=False), _is_action

        register_tool(ToolSchema(name=reg_name, description=desc,
                                 parameters=params, category=category), _executor)
        registered.append(reg_name)
    with _status_lock:
        MCP_SERVER_STATUS[name] = {"status": "connected", "tools": registered, "error": ""}
    return registered


def register_remote_tools(background: bool = True):
    """连接所有已配置的 MCP server 并注册远端工具。

    background=True 时在守护线程执行（npx 首次下载包可能较慢），
    工具注册完成后自动出现在 get_tools_for_prompt() 中。
    """
    if _registered.is_set():
        return MCP_SERVER_STATUS
    _registered.set()
    servers = discover_servers()

    def _work():
        for cfg in servers:
            try:
                _register_one_server(cfg)
            except Exception as e:
                with _status_lock:
                    MCP_SERVER_STATUS[cfg["name"]] = {"status": "failed", "tools": [], "error": str(e)}

    if background and servers:
        threading.Thread(target=_work, daemon=True, name="mcp-register").start()
    elif servers:
        _work()
    return MCP_SERVER_STATUS


def get_mcp_status() -> dict:
    """返回各 MCP server 连接状态（用于 UI 展示 / thinking 追踪）。"""
    with _status_lock:
        return json.loads(json.dumps(MCP_SERVER_STATUS))
