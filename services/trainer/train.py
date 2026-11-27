"""Trainer job: train the loan risk model and publish a new artifact version.

    python services/trainer/train.py

Steps:
  1. Load train.csv (614 past home loan applications).
  2. Compare five models with 5-fold cross-validation.
  3. Evaluate the chosen model on a 20% held-out split.
  4. Refit it on all rows and save pipeline.joblib.
  5. Write metrics.json, dashboard.json and browser_model.json.
  6. Point <ARTIFACTS_DIR>/CURRENT at the new version.

It runs, publishes and exits. Restart the scoring and insights services to load
the new version, and run services/gateway/embed.py to refresh the copy inside
the dashboard page.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from loan_features import FEATURE_LABELS, FEATURES, artifacts_root, build_features, publish

DATA_CSV = Path(__file__).parent / "train.csv"
SEED = 42


def make_pipeline(clf, scale=True):
    steps = [("features", FunctionTransformer(build_features))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


CANDIDATES = {
    "Logistic regression": make_pipeline(LogisticRegression(C=0.5, max_iter=1000)),
    "Decision tree": make_pipeline(DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=SEED), scale=False),
    "Random forest": make_pipeline(RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=SEED), scale=False),
    "Gradient boosting": make_pipeline(GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=SEED), scale=False),
    "Always approve": make_pipeline(DummyClassifier(strategy="most_frequent"), scale=False),
}
CHOSEN = "Logistic regression"


def compare_models(df, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, pipe in CANDIDATES.items():
        r = cross_validate(pipe, df, y, cv=cv, scoring=["accuracy", "roc_auc"])
        rows.append({
            "name": name,
            "accuracy": round(float(r["test_accuracy"].mean()), 3),
            "accuracy_sd": round(float(r["test_accuracy"].std()), 3),
            "auc": round(float(r["test_roc_auc"].mean()), 3),
        })
        print(f"  {name:20s} accuracy {rows[-1]['accuracy']:.3f} ± {rows[-1]['accuracy_sd']:.3f}   AUC {rows[-1]['auc']:.3f}")
    return rows


def why_chosen(comparison):
    models = [r for r in comparison if r["name"] != "Always approve"]
    best = max(models, key=lambda r: r["accuracy"])
    mine = next(r for r in models if r["name"] == CHOSEN)
    if best is mine:
        return f"Highest cross-validated accuracy of the models tried, and its weights explain every prediction."
    gap = (best["accuracy"] - mine["accuracy"]) * 100
    return (f"{best['name']} scored {gap:.1f} points higher, which is inside the fold-to-fold spread "
            f"of about {mine['accuracy_sd'] * 100:.1f} points. Logistic regression was kept because its "
            f"weights explain every prediction.") if gap <= 2 * mine["accuracy_sd"] * 100 else (
            f"{best['name']} scored {gap:.1f} points higher. Logistic regression was kept for explainability; "
            f"consider switching.")


def holdout(df, y):
    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.2, random_state=SEED, stratify=y)
    pipe = make_pipeline(LogisticRegression(C=0.5, max_iter=1000)).fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(yte, proba)
    wrong_approvals = Xte[(pred == 1) & (yte.values == 0)]
    return {
        "test_n": int(len(yte)),
        "accuracy": round(accuracy_score(yte, pred), 3),
        "precision": round(precision_score(yte, pred), 3),
        "recall": round(recall_score(yte, pred), 3),
        "auc": round(roc_auc_score(yte, proba), 3),
        "baseline": round(float(max(y.mean(), 1 - y.mean())), 3),
        "cm": confusion_matrix(yte, pred).tolist(),   # [[TN, FP], [FN, TP]]
        "roc": [[round(float(a), 3), round(float(b), 3)] for a, b in zip(fpr, tpr)],
        "wrong_approvals_clean_credit": int((wrong_approvals["Credit_History"].fillna(1) == 1).sum()),
    }


def approval_rates(df, y, col, values, labels):
    out = []
    for v, label in zip(values, labels):
        mask = df[col].isna() if v is None else df[col] == v
        out.append({"label": label, "n": int(mask.sum()), "approved": int(y[mask].sum())})
    return out


def dashboard_data(df, y):
    income = df["ApplicantIncome"] + df["CoapplicantIncome"]
    edges = [0, 3000, 4000, 5000, 6000, 8000, 10000, 15000, np.inf]
    bins = ["<3k", "3–4k", "4–5k", "5–6k", "6–8k", "8–10k", "10–15k", "15k+"]
    hist = []
    for i, label in enumerate(bins):
        m = (income >= edges[i]) & (income < edges[i + 1])
        hist.append({"label": label, "approved": int((m & (y == 1)).sum()), "rejected": int((m & (y == 0)).sum())})
    has_amount = df["LoanAmount"].notna()
    return {
        "kpi": {
            "n": int(len(df)), "approved": int(y.sum()),
            "median_income": float(income.median()), "median_loan": float(df["LoanAmount"].median()),
            "missing_rows": int(df.isnull().any(axis=1).sum()),
        },
        "by_credit": approval_rates(df, y, "Credit_History", [1.0, 0.0, None], ["Meets guidelines", "Doesn't", "Not recorded"]),
        "by_area": approval_rates(df, y, "Property_Area", ["Semiurban", "Urban", "Rural"], ["Semi-urban", "Urban", "Rural"]),
        "by_edu": approval_rates(df, y, "Education", ["Graduate", "Not Graduate"], ["Graduate", "Not graduate"]),
        "by_married": approval_rates(df, y, "Married", ["Yes", "No"], ["Married", "Single"]),
        "by_dep": approval_rates(df, y, "Dependents", ["0", "1", "2", "3+"], ["0", "1", "2", "3+"]),
        "by_self": approval_rates(df, y, "Self_Employed", ["No", "Yes"], ["Salaried", "Self-employed"]),
        "income_hist": hist,
        "scatter": [[int(i), int(a), int(s)] for i, a, s in zip(income[has_amount], df.loc[has_amount, "LoanAmount"], y[has_amount])],
        "missing": sorted(
            [{"label": k, "n": int(v)} for k, v in df.isnull().sum().items() if v > 0], key=lambda d: -d["n"]),
    }


def browser_copy(pipe):
    """The fitted numbers the page needs to score offline, when no server is running."""
    scale, clf = pipe.named_steps["scale"], pipe.named_steps["clf"]
    return {
        "features": FEATURES,
        "mean": [round(float(v), 6) for v in scale.mean_],
        "std": [round(float(v), 6) for v in scale.scale_],
        "coef": [round(float(v), 6) for v in clf.coef_[0]],
        "intercept": round(float(clf.intercept_[0]), 6),
    }


def main():
    df = pd.read_csv(DATA_CSV)
    y = (df["Loan_Status"] == "Y").astype(int)
    print(f"Loaded {len(df)} applications, {y.mean():.1%} approved")

    print("Comparing models, 5-fold cross-validation:")
    comparison = compare_models(df, y)

    print(f"Held-out test for {CHOSEN}:")
    test = holdout(df, y)
    print(f"  accuracy {test['accuracy']:.3f}  AUC {test['auc']:.3f}  (always approve: {test['baseline']:.3f})")

    final = make_pipeline(LogisticRegression(C=0.5, max_iter=1000)).fit(df, y)
    trained_at = datetime.now(timezone.utc)
    root = artifacts_root()
    version = "v" + trained_at.strftime("%Y%m%dT%H%M%S%fZ")
    out = root / version
    out.mkdir(parents=True)
    joblib.dump({
        "pipeline": final,
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
        "version": version,
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "rows": int(len(df)),
    }, out / "pipeline.joblib")

    coef = final.named_steps["clf"].coef_[0]
    cv_row = next(r for r in comparison if r["name"] == CHOSEN)
    metrics = {
        "chosen": CHOSEN,
        "why": why_chosen(comparison),
        **test,
        "cv": cv_row["accuracy"],
        "comparison": comparison,
        "weights": sorted(
            [{"key": f, "label": FEATURE_LABELS[f], "w": round(float(c), 6)} for f, c in zip(FEATURES, coef)],
            key=lambda d: -d["w"]),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    data = dashboard_data(df, y)
    data["model"] = {k: metrics[k] for k in ("accuracy", "auc", "precision", "recall", "test_n", "cm", "roc", "cv", "baseline", "wrong_approvals_clean_credit", "comparison", "chosen", "why")}
    data["weights"] = metrics["weights"]
    (out / "dashboard.json").write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    (out / "browser_model.json").write_text(json.dumps(browser_copy(final)), encoding="utf-8")

    publish(root, version)
    print(f"Published {version} to {root} (pipeline.joblib, metrics.json, dashboard.json, browser_model.json)")


if __name__ == "__main__":
    main()
