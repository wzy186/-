"""API 层验收测试：逐条映射任务书验收用例 AC-01 ~ AC-05。"""
from app.main import create_app


def test_ac01_initial_portfolio(client):
    """总资产 100,000.00、现金 20,000.00；A01 占 40%、科技占 60% 均触发；其他对象不超限。"""
    resp = client.get("/api/portfolio")
    assert resp.status_code == 200
    data = resp.json()
    assert data["portfolio_id"] == "DEMO-001"
    assert data["snapshot_at"] == "2026-11-02T15:00:00+08:00"
    assert data["analysis"]["total_assets"] == "100000.00"
    assert data["analysis"]["cash"] == "20000.00"
    assert data["analysis"]["holdings_count"] == 6

    items = {item["symbol"]: item for item in data["analysis"]["items"]}
    assert items["A01"]["weight_percent"] == "40.00"
    assert items["A01"]["single_triggered"] is True
    assert items["A01"]["industry_weight_percent"] == "60.00"
    assert items["A01"]["industry_triggered"] is True
    for symbol in ("A02", "A03", "A04", "A05", "A06"):
        assert items[symbol]["single_triggered"] is False

    conc = data["analysis"]["concentration"]
    assert [c["symbol"] for c in conc["single_instrument"]] == ["A01"]
    assert [c["industry"] for c in conc["industry"]] == ["科技"]


def test_ac02_threshold_boundary(client):
    """卖 A01×100：现金 30,000、A01 市值 30,000、总资产不变；30% 与 50% 恰好不触发。"""
    resp = client.post("/api/simulate", json={"operations": [{"action": "SELL", "symbol": "A01", "quantity": 100}]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] is None
    assert len(data["applied"]) == 1
    assert data["simulated"]["cash"] == "30000.00"
    assert data["analysis"]["total_assets"] == "100000.00"

    items = {item["symbol"]: item for item in data["analysis"]["items"]}
    assert items["A01"]["quantity"] == 300
    assert items["A01"]["market_value"] == "30000.00"
    assert items["A01"]["weight_percent"] == "30.00"
    assert items["A01"]["single_triggered"] is False
    assert items["A01"]["industry_weight_percent"] == "50.00"
    assert items["A01"]["industry_triggered"] is False
    assert data["analysis"]["concentration"]["single_instrument"] == []
    assert data["analysis"]["concentration"]["industry"] == []

    # 原始组合不受影响
    assert data["original_analysis"]["cash"] == "20000.00"
    original_items = {item["symbol"]: item for item in data["original_analysis"]["items"]}
    assert original_items["A01"]["quantity"] == 400


def test_ac03_sequential_operations(client):
    """接 AC-02 再买 A02×100：现金 25,000；A02 数量 500、市值 25,000；科技 55% 再次触发；总资产不变。"""
    ops = [
        {"action": "SELL", "symbol": "A01", "quantity": 100},
        {"action": "BUY", "symbol": "A02", "quantity": 100},
    ]
    resp = client.post("/api/simulate", json={"operations": ops})
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] is None
    assert len(data["applied"]) == 2
    assert data["simulated"]["cash"] == "25000.00"
    assert data["analysis"]["total_assets"] == "100000.00"

    items = {item["symbol"]: item for item in data["analysis"]["items"]}
    assert items["A02"]["quantity"] == 500
    assert items["A02"]["market_value"] == "25000.00"
    assert items["A02"]["industry_weight_percent"] == "55.00"
    assert items["A02"]["industry_triggered"] is True
    assert [c["industry"] for c in data["analysis"]["concentration"]["industry"]] == ["科技"]


def test_ac04_invalid_operations(client):
    """无效操作被后端拒绝并返回明确错误，不改变有效结果。"""
    cases = [
        ({"action": "BUY", "symbol": "A01", "quantity": 201}, "INSUFFICIENT_CASH", "现金"),
        ({"action": "SELL", "symbol": "A01", "quantity": 401}, "EXCESS_QUANTITY", "持仓"),
        ({"action": "BUY", "symbol": "A01", "quantity": 0}, "INVALID_QUANTITY", "正整数"),
        ({"action": "BUY", "symbol": "A01", "quantity": -5}, "INVALID_QUANTITY", "正整数"),
        ({"action": "BUY", "symbol": "A01", "quantity": 2.5}, "INVALID_QUANTITY", "正整数"),
        ({"action": "BUY", "symbol": "ZZ9", "quantity": 1}, "UNKNOWN_SYMBOL", "未知"),
    ]
    for op, code, keyword in cases:
        resp = client.post("/api/simulate", json={"operations": [op]})
        assert resp.status_code == 200, op
        data = resp.json()
        assert data["error"]["code"] == code, op
        assert data["error"]["op_index"] == 1, op
        assert keyword in data["error"]["message"], op
        # 无效操作不改变有效结果：applied 为空，模拟状态等于原始组合
        assert data["applied"] == []
        assert data["simulated"]["cash"] == "20000.00"
        holdings = {h["symbol"]: h["quantity"] for h in data["simulated"]["holdings"]}
        assert holdings["A01"] == 400


def test_ac05_persistence_and_restart(client, db_file):
    """保存 AC-03 方案；模拟服务重启后重开结果不变；原始组合不受影响。"""
    ops = [
        {"action": "SELL", "symbol": "A01", "quantity": 100},
        {"action": "BUY", "symbol": "A02", "quantity": 100},
    ]
    resp = client.post("/api/plans", json={"name": "AC-03 方案", "operations": ops})
    assert resp.status_code == 201
    plan_id = resp.json()["id"]
    assert plan_id == 1

    # “重启服务”：同一 DB 文件上重建应用与客户端
    restarted = create_app(db_path=db_file)
    from fastapi.testclient import TestClient

    client2 = TestClient(restarted)
    reopened = client2.get(f"/api/plans/{plan_id}").json()
    assert reopened["name"] == "AC-03 方案"
    assert reopened["simulated"]["cash"] == "25000.00"
    items = {item["symbol"]: item for item in reopened["analysis"]["items"]}
    assert items["A01"]["quantity"] == 300
    assert items["A02"]["quantity"] == 500
    assert items["A02"]["industry_weight_percent"] == "55.00"
    assert items["A02"]["industry_triggered"] is True

    # 原始组合仍为现金 20,000、A01 400、A02 400
    portfolio = client2.get("/api/portfolio").json()
    assert portfolio["analysis"]["cash"] == "20000.00"
    pitems = {item["symbol"]: item for item in portfolio["analysis"]["items"]}
    assert pitems["A01"]["quantity"] == 400
    assert pitems["A02"]["quantity"] == 400
