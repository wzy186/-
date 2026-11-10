"""POST /api/rebalance 接口验收测试（期望值与引擎测试同源，均经人工笔算）。"""
import pytest

from fastapi.testclient import TestClient


class TestEqualMode:
    def test_equal_weight(self, client: TestClient):
        resp = client.post("/api/rebalance", json={"mode": "equal"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is None
        assert data["suggested_ops"] == [
            {"action": "SELL", "symbol": "A01", "quantity": 233},
            {"action": "SELL", "symbol": "A02", "quantity": 67},
            {"action": "BUY", "symbol": "A03", "quantity": 67},
            {"action": "BUY", "symbol": "A04", "quantity": 233},
            {"action": "BUY", "symbol": "A05", "quantity": 137},
            {"action": "BUY", "symbol": "A06", "quantity": 146},
        ]
        # 建议经引擎校验：已全部生效（2 卖 + 4 买），现金 0，总资产不变
        assert len(data["applied"]) == 6
        assert data["simulated"]["cash"] == "0.00"
        assert data["analysis"]["total_assets"] == "100000.00"
        assert data["cash_target_percent"] == "0.00"
        # 偏差报告：6 标的 + 现金
        assert len(data["deviation"]) == 7
        cash_row = next(d for d in data["deviation"] if d["symbol"] == "CASH")
        assert cash_row["target_percent"] == "0.00"
        assert cash_row["actual_percent"] == "0.00"
        assert cash_row["deviation_percent"] == "0.00"
        a01 = next(d for d in data["deviation"] if d["symbol"] == "A01")
        assert a01["target_percent"] == "16.67"
        assert a01["actual_percent"] == "16.70"
        assert a01["deviation_percent"] == "+0.03"

    def test_equal_weight_clears_concentration(self, client: TestClient):
        data = client.post("/api/rebalance", json={"mode": "equal"}).json()
        assert data["analysis"]["concentration"]["single_instrument"] == []
        assert data["analysis"]["concentration"]["industry"] == []


class TestTargetsMode:
    def test_partial_targets(self, client: TestClient):
        resp = client.post(
            "/api/rebalance",
            json={"mode": "targets", "targets": {"A01": "0.25", "A02": "0.25"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["suggested_ops"] == [
            {"action": "SELL", "symbol": "A01", "quantity": 150},
            {"action": "BUY", "symbol": "A02", "quantity": 100},
        ]
        assert data["simulated"]["cash"] == "30000.00"
        a01 = next(d for d in data["deviation"] if d["symbol"] == "A01")
        assert a01["actual_percent"] == "25.00"
        assert a01["deviation_percent"] == "0.00"

    @pytest.mark.parametrize(
        "targets, code",
        [
            ({"A01": "0.90", "A02": "0.20"}, "INVALID_TARGETS"),  # 权重之和 > 1
            ({"ZZ9": "0.10"}, "UNKNOWN_SYMBOL"),
            ({"A01": "-0.10"}, "INVALID_TARGET"),
            ({"A01": "1.10"}, "INVALID_TARGET"),
            ({"A01": "abc"}, "INVALID_TARGET"),
            ({}, "EMPTY_TARGETS"),
        ],
    )
    def test_invalid_targets_rejected(self, client: TestClient, targets, code):
        resp = client.post("/api/rebalance", json={"mode": "targets", "targets": targets})
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == code


class TestSuggestionConsistency:
    def test_suggested_ops_pass_same_validation(self, client: TestClient):
        """建议载入调仓模拟后结果一致：同一引擎、同一口径。"""
        suggestion = client.post("/api/rebalance", json={"mode": "equal"}).json()
        resp = client.post("/api/simulate", json={"operations": suggestion["suggested_ops"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is None
        assert len(data["applied"]) == len(suggestion["applied"])
        assert data["simulated"]["cash"] == suggestion["simulated"]["cash"]
