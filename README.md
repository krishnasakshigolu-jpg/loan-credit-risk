# Loan Credit Risk Predictor

Predicts how likely a home loan application is to be approved, explains which details pushed the score up or down, and wraps it all in a dashboard with data insights, model evaluation and loan calculators.

AI/ML Lab project, 5th semester, B.Tech CSE (Robotics and IoT), Bennett University.

## What's in it

- **Model:** a scikit-learn pipeline (feature engineering, scaling, logistic regression) trained on `train.csv`.
- **Services:** a trainer job and three small FastAPI services (scoring, insights and a gateway), each running as its own local process. See [Architecture](#architecture).
- **Dashboard:** `services/gateway/static/index.html`, with four sections:
  - **Assess:** enter an application and get an approval chance, a risk band, the top factors behind the score and a what-if chart across loan amounts.
  - **Insights:** approval rates by credit history, property area, marital status, education, employment and dependents. Also an income histogram, a loan-against-income scatter, and missing data per column.
  - **Model:** cross-validated and held-out accuracy, confusion matrix, ROC curve, feature weights and the model comparison.
  - **Calculators:** EMI with a yearly principal and interest split, affordability, and prepayment savings.

## Architecture

```
  Browser ──▶ gateway :8000 ──┬──▶ scoring  :8001   /api/predict, /api/what-if, /api/health
             (serves the page)└──▶ insights :8002   /api/stats
                                        │ both load the CURRENT version at startup
                                        ▼
                                  artifacts/  ◀── trainer (runs, publishes, exits)
```

| Service | Folder | Owns | Needs scikit-learn |
|---|---|---|---|
| trainer | `services/trainer` | `train.csv`, model selection, evaluation | yes |
| scoring | `services/scoring` | the fitted pipeline, predictions and explanations | yes |
| insights | `services/insights` | the data behind the Insights and Model pages | no |
| gateway | `services/gateway` | the dashboard page and routing `/api/*` to the owning service | no |
| shared library | `libs/loan_features` | features, the request schema, artifact loading | no |

- **Artifacts are versioned.** Each training run writes a new `artifacts/v<timestamp>/` folder (pipeline, metrics, dashboard data, browser copy) and only then points `artifacts/CURRENT` at it, so a service never loads a half-written model. Set `ARTIFACTS_DIR` to keep them elsewhere.
- **The page doesn't know there are several services.** It still calls relative `/api/...` paths, and the gateway forwards each one. `GET /healthz` on the gateway reports whether every service behind it is up.
- **One set of pinned versions.** `constraints.txt` pins everything. The trainer and the scoring service must run the same scikit-learn, or the saved pipeline won't load. The pipeline also refers to `loan_features.features` by name, which is why that code lives in a shared package.
- **Everything runs locally.** No Docker, no containers. `scripts/dev.py` starts each service as a separate Python process.

## Dashboard design

The dashboard follows Apple's design language: system colours, grouped forms, segmented controls, and Liquid Glass only on the sidebar, tab bar and top bar. It is one HTML file with no frameworks, fonts or CDN calls, so it also works opened straight from disk.

- Works at phone, tablet and desktop widths, in light and dark mode.
- Every chart can be read by keyboard: Tab to it, then use the arrow keys. Each value shows a tooltip and is announced to screen readers. Every chart also has a data table.
- Text meets 4.5:1 contrast, touch controls are at least 44 px, and the page respects reduced motion and reduced transparency settings.
- Every number in the page text comes from the data or the model, so retraining updates the page.

## Dataset

`train.csv` has 614 past applications with 13 columns: applicant and co-applicant monthly income, loan amount (in thousands), term (in months), credit history, property area, marital status, dependents, education, self-employment, gender and the outcome (`Loan_Status`). 422 applications (68.7%) were approved. 134 rows have at least one blank value.

## Features

`features.py` turns each row into 10 numbers:

| Feature | How it's built |
|---|---|
| credit_history | 1 if it meets guidelines. Blanks count as 1, since 74% of those applicants were approved. |
| married, graduate, self_employed | Yes/no as 1/0 |
| semiurban, rural | Property area, with urban as the reference |
| dependents | 0 to 3, where 3 means three or more |
| log_total_income | log(1 + applicant + co-applicant income) |
| log_loan_amount | log(1 + loan amount). Blanks use the median, 128. |
| emi_to_income | (amount x 1000 / term) / household income |

Gender is left out on purpose. A lending model shouldn't score people on it.

## Results

5-fold cross-validation on all 614 rows:

| Model | Accuracy | ROC AUC |
|---|---|---|
| Logistic regression | 80.3% ± 1.7 | 0.757 |
| Decision tree | 79.8% ± 2.5 | 0.722 |
| Random forest | 80.9% ± 1.2 | 0.777 |
| Gradient boosting | 79.8% ± 2.0 | 0.768 |
| Always approve | 68.7% ± 0.4 | 0.500 |

Random forest scored 0.6 points higher, which is inside the fold-to-fold spread of about 1.7 points. Logistic regression was kept because its weights explain every prediction.

On a 20% held-out split (123 applications), the served model scores 85.4% accuracy, AUC 0.863, precision 83.2% and recall 98.8%. That beats the cross-validated figure because this split happens to be an easier one. Treat 80.3% as the honest estimate.

Confusion matrix on the held-out split:

|  | Predicted reject | Predicted approve |
|---|---|---|
| Actually rejected | 21 | 17 |
| Actually approved | 1 | 84 |

The model almost never turns away a good applicant, but it lets through 17 of 38 risky ones. All 17 of those had a clean credit history, which is by far the strongest signal in the data (80% approval with it, 8% without).

## Run it

Needs Python 3.10 or newer. Versions are pinned in `constraints.txt` (scikit-learn 1.5.2), tested on Python 3.12 and 3.13.

On Windows, the quickest way is to double-click `run.bat`. The first run creates `.venv` and installs the requirements. It then trains the model if none is published, starts every service and opens the dashboard in your browser. Close its window to stop them all.

Or do the same steps by hand:

```
python3 -m venv .venv
.venv\Scripts\activate
pip3 install -r requirements.txt
python3 scripts/dev.py
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

Open http://127.0.0.1:8000. The page shows "Scored by the Python model through FastAPI" when it's connected. `python scripts/dev.py --train` retrains before starting.

To run one service on its own, for example while working on it, start it from its folder:

```
python services/trainer/train.py                                              # publish a model
cd services/scoring  && uvicorn scoring_service.main:app --port 8001 --reload
cd services/insights && uvicorn insights_service.main:app --port 8002 --reload
cd services/gateway  && uvicorn gateway_service.main:app --port 8000 --reload  # SCORING_URL / INSIGHTS_URL to point elsewhere
```

Scoring and insights load the model when they start, so restart them after retraining.

You can also open `services/gateway/static/index.html` directly with no services running. It falls back to a copy of the model built into the page, so the dashboard still works offline. After retraining, run `python services/gateway/embed.py` to refresh that copy. A test checks that it matches the scoring service.

Run the tests with `python -m pytest -q` (train first). Each service has its own folder under `tests/`, and `tests/integration` wires the gateway to the real services.

## Put it online

`render.yaml` deploys the whole stack to [Render](https://render.com) as one free web service. It runs `scripts/dev.py`, so the services are separate processes on the same machine and only the gateway is public. In Render, choose **New**, then **Blueprint**, and pick this repository. The build installs the requirements and trains the model, then starts the services. Each push to `main` deploys again. Free services sleep when idle, so the first visit after a quiet spell takes about half a minute.

## API

All routes go through the gateway on port 8000. Each service has its own interactive docs at `/docs` on its port: scoring on 8001, insights on 8002.

| Method | Path | Service | What it does |
|---|---|---|---|
| GET | `/healthz` | gateway | Whether every service is up (503 if not) |
| GET | `/api/health` | scoring | Model name, version, training time, row count |
| GET | `/api/stats` | insights | Data behind the Insights and Model pages |
| POST | `/api/predict` | scoring | Scores one application |
| POST | `/api/what-if` | scoring | Approval chance across loan amounts from ₹10k to ₹700k |

Example request:

```json
{
  "applicant_income": 4583,
  "coapplicant_income": 1508,
  "loan_amount": 128,
  "loan_term": 360,
  "credit_history": 1,
  "property_area": "Rural",
  "married": true,
  "dependents": 1,
  "graduate": true,
  "self_employed": false
}
```

The response includes `probability`, `band` (low, moderate or high), and `factors`, each feature's push on the log-odds, largest first.

## Files

```
libs/loan_features/          shared package: features, request schema, artifact loading
services/trainer/            train.py, train.csv, explore.py
services/scoring/            scoring service
services/insights/           insights service
services/gateway/            gateway, static/index.html (dashboard), embed.py
scripts/dev.py               starts every service locally
tests/                       one folder per service, plus integration
artifacts/                   published model versions (created by training, not committed)
constraints.txt              pinned versions for every service
requirements.txt             installs everything for local runs and tests
run.bat                      one-click setup and start on Windows
render.yaml                  deploy settings for Render
```

## Limitations

- 614 rows is small. Scores shift by a point or two between random splits.
- Credit history dominates. Income barely moves the score, so the model can't tell a stretched applicant from a comfortable one if both have clean credit.
- Loan amounts outside ₹9k to ₹700k are beyond what the model has seen, and the API flags them.
- The repayment the model sees is amount divided by term, without interest. The EMI calculator adds interest.
- This is a lab project, not a lending tool.

## Credits

Frontend designed by Harshit Poonia.
