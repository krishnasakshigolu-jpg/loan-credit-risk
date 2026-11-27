"""Insights service on its own."""
from fastapi.testclient import TestClient

from insights_service.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_stats_match_dataset():
    d = client.get("/api/stats").json()
    assert d["kpi"]["n"] == 614
    assert d["kpi"]["approved"] == 422
    assert {m["name"] for m in d["model"]["comparison"]} >= {"Logistic regression", "Random forest", "Always approve"}
