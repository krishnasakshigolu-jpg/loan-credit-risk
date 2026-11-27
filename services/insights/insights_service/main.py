"""Insights service: the data behind the dashboard's Insights and Model pages.

    uvicorn insights_service.main:app --port 8002   (from services/insights)

Read-only. Serves dashboard.json from the CURRENT artifact version and does
not need scikit-learn.
"""
import json

from fastapi import FastAPI

from loan_features import current_dir

MODEL_DIR = current_dir()
DASHBOARD = json.loads((MODEL_DIR / "dashboard.json").read_text(encoding="utf-8"))

app = FastAPI(title="Loan Risk insights service", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "insights", "version": MODEL_DIR.name}


@app.get("/api/stats")
def stats():
    """Everything the Insights and Model pages draw."""
    return DASHBOARD
