"""FastAPI server for the Loan Risk dashboard.

    uvicorn app:app --reload
    then open http://127.0.0.1:8000

Serves index.html and a small JSON API backed by model/pipeline.joblib.
"""
import json
import warnings
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from features import FEATURES

ROOT = Path(__file__).parent
MODEL_FILE = ROOT / "model" / "pipeline.joblib"
DASHBOARD_FILE = ROOT / "model" / "dashboard.json"
METRICS_FILE = ROOT / "model" / "metrics.json"

if not MODEL_FILE.exists():
    raise SystemExit("model/pipeline.joblib is missing. Run `python train.py` first.")

try:
    BUNDLE = joblib.load(MODEL_FILE)
except Exception as exc:  # usually a scikit-learn version mismatch
    raise SystemExit(f"Could not load model/pipeline.joblib ({exc}). Run `python train.py` to rebuild it.")
if BUNDLE["sklearn_version"] != sklearn.__version__:
    warnings.warn(f"Model was saved with scikit-learn {BUNDLE['sklearn_version']} but {sklearn.__version__} "
                  "is installed. Run `python train.py` to rebuild it.")
PIPE = BUNDLE["pipeline"]
METRICS = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
DASHBOARD = json.loads(DASHBOARD_FILE.read_text(encoding="utf-8"))
LABELS = {w["key"]: w["label"] for w in METRICS["weights"]}

# Loan amounts seen in training, in thousands of rupees.
TRAINED_RANGE = (9, 700)

app = FastAPI(title="Loan Risk API", version="1.0.0",
              description="Approval chance for a home loan application, from a logistic regression trained on train.csv.")


class Application(BaseModel):
    applicant_income: float = Field(..., gt=0, le=1_000_000, description="Monthly income, rupees")
    coapplicant_income: float = Field(0, ge=0, le=1_000_000, description="Monthly income, rupees")
    loan_amount: float = Field(..., gt=0, le=10_000, description="Thousands of rupees")
    loan_term: int = Field(360, gt=0, le=600, description="Months")
    credit_history: Optional[Literal[0, 1]] = Field(None, description="1 meets guidelines, 0 does not, null unknown")
    property_area: Literal["Urban", "Semiurban", "Rural"] = "Urban"
    married: bool = False
    dependents: int = Field(0, ge=0, le=3, description="3 means three or more")
    graduate: bool = True
    self_employed: bool = False

    model_config = {"json_schema_extra": {"examples": [{
        "applicant_income": 4583, "coapplicant_income": 1508, "loan_amount": 128, "loan_term": 360,
        "credit_history": 1, "property_area": "Rural", "married": True, "dependents": 1,
        "graduate": True, "self_employed": False}]}}


def to_rows(app_: Application, amounts=None) -> pd.DataFrame:
    """One row per loan amount, in the same columns as train.csv."""
    amounts = [app_.loan_amount] if amounts is None else amounts
    base = {
        "ApplicantIncome": app_.applicant_income,
        "CoapplicantIncome": app_.coapplicant_income,
        "Loan_Amount_Term": float(app_.loan_term),
        "Credit_History": np.nan if app_.credit_history is None else float(app_.credit_history),
        "Property_Area": app_.property_area,
        "Married": "Yes" if app_.married else "No",
        "Dependents": "3+" if app_.dependents >= 3 else str(app_.dependents),
        "Education": "Graduate" if app_.graduate else "Not Graduate",
        "Self_Employed": "Yes" if app_.self_employed else "No",
    }
    return pd.DataFrame([{**base, "LoanAmount": float(a)} for a in amounts])


def band(p: float) -> str:
    return "low" if p >= 0.70 else "moderate" if p >= 0.45 else "high"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": METRICS["chosen"], "trained_at": BUNDLE["trained_at"],
            "rows": BUNDLE["rows"], "sklearn_version": BUNDLE["sklearn_version"]}


@app.get("/api/stats")
def stats():
    """Everything the Insights and Model pages draw."""
    return DASHBOARD


@app.post("/api/predict")
def predict(app_: Application):
    rows = to_rows(app_)
    p = float(PIPE.predict_proba(rows)[0, 1])

    # Each feature's push on the log-odds: weight x standardised value.
    z = PIPE.named_steps["scale"].transform(PIPE.named_steps["features"].transform(rows))[0]
    coef = PIPE.named_steps["clf"].coef_[0]
    factors = sorted(
        [{"key": f, "label": LABELS[f], "effect": round(float(c * v), 4)} for f, c, v in zip(FEATURES, coef, z)],
        key=lambda d: -abs(d["effect"]))

    income = app_.applicant_income + app_.coapplicant_income
    repayment = app_.loan_amount * 1000 / app_.loan_term
    return {
        "probability": round(p, 4),
        "band": band(p),
        "factors": factors,
        "household_income": income,
        "monthly_repayment": round(repayment, 2),
        "repayment_share": round(repayment / income, 4),
        "credit_history_assumed": app_.credit_history is None,
        "outside_training_range": not (TRAINED_RANGE[0] <= app_.loan_amount <= TRAINED_RANGE[1]),
    }


@app.post("/api/what-if")
def what_if(app_: Application, start: int = 10, stop: int = 700, step: int = 5):
    """Approval chance across loan amounts, everything else held the same."""
    if not (0 < start < stop) or step <= 0 or (stop - start) / step > 1000:
        raise HTTPException(422, "Use 0 < start < stop and at most 1000 steps.")
    amounts = list(range(start, stop + 1, step))
    probs = PIPE.predict_proba(to_rows(app_, amounts))[:, 1]
    return {"points": [[a, round(float(p), 4)] for a, p in zip(amounts, probs)]}
