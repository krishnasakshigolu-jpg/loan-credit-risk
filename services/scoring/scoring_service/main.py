"""Scoring service: approval chance and explanation for one application.

    uvicorn scoring_service.main:app --port 8001   (from services/scoring)

Loads the pipeline from the CURRENT artifact version at startup. Stateless,
so several copies can run behind the gateway.
"""
import json
import warnings

import joblib
import sklearn
from fastapi import FastAPI, HTTPException

from loan_features import FEATURES, TRAINED_RANGE, Application, band, current_dir, to_rows

MODEL_DIR = current_dir()
try:
    BUNDLE = joblib.load(MODEL_DIR / "pipeline.joblib")
except Exception as exc:  # usually a scikit-learn version mismatch
    raise SystemExit(f"Could not load {MODEL_DIR / 'pipeline.joblib'} ({exc}). Retrain with the installed scikit-learn.")
if BUNDLE["sklearn_version"] != sklearn.__version__:
    warnings.warn(f"Model was saved with scikit-learn {BUNDLE['sklearn_version']} but {sklearn.__version__} "
                  "is installed. Retrain to rebuild it.")
PIPE = BUNDLE["pipeline"]
METRICS = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
LABELS = {w["key"]: w["label"] for w in METRICS["weights"]}

app = FastAPI(title="Loan Risk scoring service", version="1.0.0",
              description="Approval chance for a home loan application, from a logistic regression trained on train.csv.")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "scoring", "model": METRICS["chosen"], "version": BUNDLE["version"],
            "trained_at": BUNDLE["trained_at"], "rows": BUNDLE["rows"], "sklearn_version": BUNDLE["sklearn_version"]}


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
