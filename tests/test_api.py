"""Run with: python -m pytest -q"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402

client = TestClient(app)

# LP001014 from train.csv: no qualifying credit history, actually rejected.
LP001014 = {"applicant_income": 3036, "coapplicant_income": 2504, "loan_amount": 158, "loan_term": 360,
            "credit_history": 0, "property_area": "Semiurban", "married": True, "dependents": 3,
            "graduate": True, "self_employed": False}


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["rows"] == 614


def test_index_is_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Loan Risk" in r.text


def test_stats_match_dataset():
    d = client.get("/api/stats").json()
    assert d["kpi"]["n"] == 614
    assert d["kpi"]["approved"] == 422
    assert {m["name"] for m in d["model"]["comparison"]} >= {"Logistic regression", "Random forest", "Always approve"}


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


def test_browser_copy_matches_server():
    """The offline copy embedded in index.html must give the same answer as the API."""
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    m = json.loads(re.search(r"/\*MODEL\*/(.*?)/\*END\*/", html, re.S).group(1))
    a = LP001014
    income = a["applicant_income"] + a["coapplicant_income"]
    x = {"credit_history": 0, "married": 1, "graduate": 1, "self_employed": 0, "semiurban": 1, "rural": 0,
         "dependents": 3, "log_total_income": math.log1p(income), "log_loan_amount": math.log1p(a["loan_amount"]),
         "emi_to_income": a["loan_amount"] * 1000 / a["loan_term"] / income}
    z = m["intercept"] + sum(c * (x[f] - mu) / sd for f, c, mu, sd in zip(m["features"], m["coef"], m["mean"], m["std"]))
    browser = 1 / (1 + math.exp(-z))
    server = client.post("/api/predict", json=a).json()["probability"]
    assert abs(browser - server) < 1e-3
