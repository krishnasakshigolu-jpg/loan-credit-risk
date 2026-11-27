"""Scoring service on its own."""
from fastapi.testclient import TestClient

from conftest import LP001014
from scoring_service.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["rows"] == 614


def test_bad_credit_is_high_risk():
    d = client.post("/api/predict", json=LP001014).json()
    assert d["band"] == "high"
    assert d["probability"] < 0.45
    assert d["factors"][0]["key"] == "credit_history"
    assert d["factors"][0]["effect"] < 0


def test_clean_credit_scores_higher():
    bad = client.post("/api/predict", json=LP001014).json()["probability"]
    good = client.post("/api/predict", json={**LP001014, "credit_history": 1}).json()["probability"]
    assert good > bad + 0.4


def test_unknown_credit_is_flagged():
    d = client.post("/api/predict", json={**LP001014, "credit_history": None}).json()
    assert d["credit_history_assumed"] is True


def test_out_of_range_amount_is_flagged():
    d = client.post("/api/predict", json={**LP001014, "loan_amount": 900}).json()
    assert d["outside_training_range"] is True


def test_invalid_input_is_rejected():
    assert client.post("/api/predict", json={**LP001014, "applicant_income": -5}).status_code == 422
    assert client.post("/api/predict", json={**LP001014, "property_area": "Mars"}).status_code == 422
    assert client.post("/api/predict", json={**LP001014, "dependents": 7}).status_code == 422


def test_what_if_curve():
    pts = client.post("/api/what-if", json=LP001014).json()["points"]
    assert pts[0][0] == 10 and pts[-1][0] == 700
    assert all(0 <= p <= 1 for _, p in pts)
    # Bigger loans should never raise the approval chance.
    assert all(b[1] <= a[1] + 1e-9 for a, b in zip(pts, pts[1:]))
