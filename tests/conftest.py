"""Shared test setup. Services load the published model at import time, so a
trained model has to exist first:  python services/trainer/train.py
"""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ARTIFACTS_DIR", str(ROOT / "artifacts"))
if not (Path(os.environ["ARTIFACTS_DIR"]) / "CURRENT").exists():
    pytest.exit("No model published. Run `python services/trainer/train.py` before the tests.", returncode=1)

# LP001014 from train.csv: no qualifying credit history, actually rejected.
LP001014 = {"applicant_income": 3036, "coapplicant_income": 2504, "loan_amount": 158, "loan_term": 360,
            "credit_history": 0, "property_area": "Semiurban", "married": True, "dependents": 3,
            "graduate": True, "self_employed": False}
