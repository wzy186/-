"""模拟预览与方案持久化接口（FR-03 / FR-04 / FR-05）。

错误约定：
- 业务错误统一 {"detail": {"code": ..., "message": ..., ...}}，message 为中文；
- POST /api/simulate 的操作级失败返回 200 + 内嵌 error：有效前缀状态本身就是
  前端需要的输出（"校验失败不得改变有效模拟结果"）；
- POST /api/plans 在保存前重新完整校验，失败 422 且不落库。
"""
import json
import sqlite3

from fastapi import APIRouter, HTTPException, Request

from .. import db
from ..engine import analyze, parse_portfolio, simulate_sequential
from ..presenters import analysis_payload, simulate_payload
from ..schemas import PlanCreate, SimulateRequest

router = APIRouter(prefix="/api", tags=["plans"])

MAX_NAME_LENGTH = 50


def _engine(request: Request):
    state, instruments, rules = request.app.state.portfolio
    return state, instruments, rules


@router.post("/simulate")
def simulate(request: Request, body: SimulateRequest) -> dict:
    """模拟预览：从原始组合出发逐笔执行，不落库。"""
    state, instruments, rules = _engine(request)
    result = simulate_sequential(
        state,
        [(op.action, op.symbol, op.quantity) for op in body.operations],
        instruments,
    )
    return simulate_payload(result, instruments, rules, analyze(state, instruments, rules))


@router.post("/plans", status_code=201)
def create_plan(request: Request, body: PlanCreate) -> dict:
    """保存方案：校验通过后写入 SQLite；重名 409，非法输入 422。"""
    name = body.name.strip()
    if not name or len(name) > MAX_NAME_LENGTH:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_NAME", "message": f"方案名称必填且不超过 {MAX_NAME_LENGTH} 字"},
        )
    if not body.operations:
        raise HTTPException(
            status_code=422,
            detail={"code": "EMPTY_PLAN", "message": "方案至少包含一笔操作"},
        )

    state, instruments, rules = _engine(request)
    result = simulate_sequential(
        state,
        [(op.action, op.symbol, op.quantity) for op in body.operations],
        instruments,
    )
    if result.error is not None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": result.error.code,
                "message": result.error.message,
                "op_index": result.error.op_index,
            },
        )

    created_at = db.now_iso()
    try:
        plan_id = db.insert_plan(
            request.app.state.db_path,
            name,
            state.snapshot_id,
            json.dumps([{"action": op.operation.action, "symbol": op.operation.symbol, "quantity": op.operation.quantity}
                        for op in result.applied], ensure_ascii=False),
            created_at,
        )
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail={"code": "DUPLICATE_NAME", "message": "方案名称已存在，请换一个名称"},
        )
    return {
        "id": plan_id,
        "name": name,
        "created_at": created_at,
        "snapshot_id": state.snapshot_id,
        "operations": [{"action": op.operation.action, "symbol": op.operation.symbol, "quantity": op.operation.quantity}
                       for op in result.applied],
    }


@router.get("/plans")
def list_plans(request: Request) -> dict:
    return {"plans": db.list_plans(request.app.state.db_path)}


@router.delete("/plans/{plan_id}", status_code=204)
def delete_plan(request: Request, plan_id: int) -> None:
    """删除方案（前端二次确认；删除后不可恢复）。"""
    if db.get_plan(request.app.state.db_path, plan_id) is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "PLAN_NOT_FOUND", "message": "方案不存在"},
        )
    db.delete_plan(request.app.state.db_path, plan_id)


@router.get("/plans/{plan_id}")
def get_plan(request: Request, plan_id: int) -> dict:
    """重开方案：按库内快照重算模拟结果（操作列表为唯一事实来源）。"""
    row = db.get_plan(request.app.state.db_path, plan_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "PLAN_NOT_FOUND", "message": "方案不存在"},
        )
    payload = db.get_snapshot_payload(request.app.state.db_path, row["snapshot_id"])
    if payload is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "SNAPSHOT_MISSING", "message": "方案关联的快照缺失"},
        )
    base, instruments, rules = parse_portfolio(payload)
    operations = json.loads(row["ops_json"])
    result = simulate_sequential(
        base,
        [(op["action"], op["symbol"], op["quantity"]) for op in operations],
        instruments,
    )
    simulated = simulate_payload(result, instruments, rules, analyze(base, instruments, rules))
    return {
        "id": row["id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "snapshot_id": row["snapshot_id"],
        "operations": operations,
        **simulated,
    }
