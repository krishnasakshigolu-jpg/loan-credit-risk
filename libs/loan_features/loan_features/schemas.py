"""The scoring request and how it maps onto train.csv columns."""
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Loan amounts seen in training, in thousands of rupees.
TRAINED_RANGE = (9, 700)


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
