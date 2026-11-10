"""FastAPI 应用入口。

- /api/* 由 routers 提供（先注册，优先匹配）；
- / 及静态资源由 frontend/dist 托管（若已构建）；
- 未知 /api 路径返回 JSON 404；
- 启动时初始化 SQLite（建表 + 写入种子快照），引擎解析结果挂到 app.state。
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from . import config, db
from .engine import parse_portfolio
from .routers import plans, portfolio, rebalance


def create_app(db_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="投资组合分析与调仓模拟", version="0.1.0")

    db_path = db_path or config.DB_PATH
    seed_payload = json.loads(config.SEED_FILE.read_text(encoding="utf-8"))
    db.init_db(db_path, seed_payload)
    app.state.db_path = db_path
    app.state.portfolio = parse_portfolio(seed_payload)

    app.include_router(portfolio.router)
    app.include_router(plans.router)
    app.include_router(rebalance.router)

    @app.get("/api/{full_path:path}", include_in_schema=False)
    def api_not_found(full_path: str):
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"接口不存在：/api/{full_path}"},
        )

    if config.DIST_DIR.is_dir():
        app.mount("/", StaticFiles(directory=config.DIST_DIR, html=True), name="frontend")

    return app


app = create_app()
