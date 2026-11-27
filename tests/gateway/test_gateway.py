"""Gateway behaviour that doesn't need the real services."""
import httpx
import pytest
from fastapi.testclient import TestClient

from gateway_service.main import app


@pytest.fixture
def downstream():
    """Fake services that record what the gateway forwarded."""
    seen = []

    def handler(request: httpx.Request):
        seen.append((request.url.host, request.method, request.url.path, request.url.query.decode(), request.content))
        return httpx.Response(200, json={"from": request.url.host})

    saved = app.state.clients
    app.state.clients = {name: httpx.AsyncClient(base_url=f"http://{name}", transport=httpx.MockTransport(handler))
                         for name in saved}
    yield seen
    app.state.clients = saved


def test_index_is_served():
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert "Loan Risk" in r.text


@pytest.mark.parametrize("method,path,service", [
    ("GET", "health", "scoring"), ("POST", "predict", "scoring"),
    ("POST", "what-if", "scoring"), ("GET", "stats", "insights"),
])
def test_routes_go_to_owning_service(downstream, method, path, service):
    r = TestClient(app).request(method, f"/api/{path}", json={} if method == "POST" else None)
    assert r.json() == {"from": service}
    assert downstream[0][:3] == (service, method, f"/api/{path}")


def test_query_and_body_are_forwarded(downstream):
    TestClient(app).post("/api/what-if?start=20&stop=100", json={"a": 1})
    assert downstream[0][3] == "start=20&stop=100"
    assert downstream[0][4] == b'{"a":1}'


def test_unknown_route_and_wrong_method(downstream):
    c = TestClient(app)
    assert c.get("/api/nope").status_code == 404
    assert c.get("/api/predict").status_code == 405
    assert downstream == []


def test_unreachable_service_gives_502():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    saved = app.state.clients
    app.state.clients = {n: httpx.AsyncClient(base_url=f"http://{n}", transport=httpx.MockTransport(refuse)) for n in saved}
    try:
        c = TestClient(app)
        assert c.post("/api/predict", json={}).status_code == 502
        h = c.get("/healthz")
        assert h.status_code == 503
        assert h.json()["services"] == {"scoring": "unreachable", "insights": "unreachable"}
    finally:
        app.state.clients = saved
