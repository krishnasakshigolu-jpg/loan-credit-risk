"""Gateway: the one address the browser talks to.

    uvicorn gateway_service.main:app --port 8000   (from services/gateway)

Serves the dashboard page and forwards each /api/* route to the service that
owns it, so the page keeps using same-origin relative paths.
"""
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse

STATIC = Path(__file__).resolve().parents[1] / "static"

SERVICES = {
    "scoring": os.environ.get("SCORING_URL", "http://127.0.0.1:8001"),
    "insights": os.environ.get("INSIGHTS_URL", "http://127.0.0.1:8002"),
}
# Path under /api/ -> (service, allowed methods)
ROUTES = {
    "health": ("scoring", {"GET"}),
    "predict": ("scoring", {"POST"}),
    "what-if": ("scoring", {"POST"}),
    "stats": ("insights", {"GET"}),
}
# Each service's own liveness endpoint, checked by /healthz.
LIVENESS = {"scoring": "/api/health", "insights": "/health"}

app = FastAPI(title="Loan Risk gateway", version="1.0.0")
# One client per service. Tests swap these for in-process clients.
app.state.clients = {name: httpx.AsyncClient(base_url=url, timeout=10) for name, url in SERVICES.items()}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
async def healthz():
    """Up only when every service behind the gateway answers."""
    status = {}
    for name, client in app.state.clients.items():
        try:
            status[name] = "ok" if (await client.get(LIVENESS[name])).status_code == 200 else "error"
        except httpx.HTTPError:
            status[name] = "unreachable"
    ok = all(v == "ok" for v in status.values())
    return JSONResponse({"status": "ok" if ok else "degraded", "services": status}, status_code=200 if ok else 503)


@app.api_route("/api/{path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    if path not in ROUTES:
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    service, methods = ROUTES[path]
    if request.method not in methods:
        return JSONResponse({"detail": "Method Not Allowed"}, status_code=405)
    try:
        r = await app.state.clients[service].request(
            request.method, f"/api/{path}", params=request.query_params, content=await request.body(),
            headers={k: v for k, v in request.headers.items() if k.lower() == "content-type"})
    except httpx.HTTPError:
        return JSONResponse({"detail": f"The {service} service is not reachable."}, status_code=502)
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
