"""共享 fixture：种子数据加载、组合解析与测试应用实例。"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.engine import InstrumentDef, PortfolioState, Rules, parse_portfolio
from app.main import create_app

SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "投资组合示例数据.json"


@pytest.fixture(scope="session")
def seed_payload() -> dict:
    with open(SEED_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def base_portfolio(seed_payload) -> tuple[PortfolioState, dict[str, InstrumentDef], Rules]:
    return parse_portfolio(seed_payload)


@pytest.fixture()
def db_file(tmp_path) -> Path:
    """每个测试独立的临时数据库文件。"""
    return tmp_path / "test.db"


@pytest.fixture()
def app(db_file):
    return create_app(db_path=db_file)


@pytest.fixture()
def client(app) -> TestClient:
    return TestClient(app)
