"""Feature engineering shared by training and the API.

Takes rows shaped like train.csv and turns them into the 10 numeric
features the model uses. Gender and Loan_ID are dropped on purpose.
"""
import numpy as np
import pandas as pd

FEATURES = [
    "credit_history",
    "married",
    "graduate",
    "self_employed",
    "semiurban",
    "rural",
    "dependents",
    "log_total_income",
    "log_loan_amount",
    "emi_to_income",
]

# Fill values for blanks, taken from the training data.
MEDIAN_LOAN_AMOUNT = 128.0   # thousands
DEFAULT_TERM = 360.0         # months, by far the most common term


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    # A blank credit history is treated as "meets guidelines": 74% of those
    # applicants were approved, close to the 80% for a recorded clean history.
    X["credit_history"] = df["Credit_History"].fillna(1.0).astype(float)
    X["married"] = (df["Married"] == "Yes").astype(float)
    X["graduate"] = (df["Education"] == "Graduate").astype(float)
    X["self_employed"] = (df["Self_Employed"] == "Yes").astype(float)
    X["semiurban"] = (df["Property_Area"] == "Semiurban").astype(float)
    X["rural"] = (df["Property_Area"] == "Rural").astype(float)
    X["dependents"] = (
        df["Dependents"].astype("string").str.replace("3+", "3", regex=False).fillna("0").astype(float)
    )

    income = (df["ApplicantIncome"] + df["CoapplicantIncome"]).astype(float)
    amount = df["LoanAmount"].fillna(MEDIAN_LOAN_AMOUNT).astype(float)
    term = df["Loan_Amount_Term"].fillna(DEFAULT_TERM).astype(float)

    X["log_total_income"] = np.log1p(income)
    X["log_loan_amount"] = np.log1p(amount)
    # Monthly repayment before interest, as a share of household income.
    X["emi_to_income"] = (amount * 1000 / term) / income.clip(lower=1)
    return X[FEATURES]
