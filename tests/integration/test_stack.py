"""Gateway wired to the real scoring and insights apps, in process."""
import json
import math
import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import LP001014
from gateway_service.main import app as gateway
from insights_service.main import app as insights
from scoring_service.main import app as scoring

INDEX = Path(__file__).resolve().parents[2] / "services" / "gateway" / "static" / "index.html"


@pytest.fixture(scope="module")
def client():
    saved = gateway.state.clients
    gateway.state.clients = {
        "scoring": httpx.AsyncClient(base_url="http://scoring", transport=httpx.ASGITransport(app=scoring)),
        "insights": httpx.AsyncClient(base_url="http://insights", transport=httpx.ASGITransport(app=insights)),
    }
    yield TestClient(gateway)
    gateway.state.clients = saved


def test_healthz_all_up(client):
    assert client.get("/healthz").json() == {"status": "ok", "services": {"scoring": "ok", "insights": "ok"}}


def test_dashboard_calls_work_through_gateway(client):
    assert client.get("/api/health").json()["rows"] == 614
    assert client.get("/api/stats").json()["kpi"]["n"] == 614
    assert client.post("/api/predict", json=LP001014).json()["band"] == "high"
    assert len(client.post("/api/what-if?step=10", json=LP001014).json()["points"]) == 70
    assert client.post("/api/predict", json={**LP001014, "dependents": 7}).status_code == 422


def test_browser_copy_matches_server(client):
    """The offline copy embedded in index.html must give the same answer as the scoring service."""
    html = INDEX.read_text(encoding="utf-8")
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
