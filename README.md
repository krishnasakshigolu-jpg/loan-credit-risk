# Loan Credit Risk Predictor

Predicts how likely a home loan application is to be approved, explains which details pushed the score up or down, and wraps it all in a dashboard with data insights, model evaluation and loan calculators.

AI/ML Lab project, 5th semester, B.Tech CSE (Robotics and IoT), Bennett University.

## What's in it

- **Model:** a scikit-learn pipeline (feature engineering, scaling, logistic regression) trained on `train.csv`.
- **API:** FastAPI server that loads the saved pipeline and scores applications.
- **Dashboard:** `index.html`, with four sections:
  - **Assess:** enter an application and get an approval chance, a risk band, the top factors behind the score and a what-if chart across loan amounts.
  - **Insights:** approval rates by credit history, property area, marital status, education, employment and dependents. Also an income histogram, a loan-against-income scatter, and missing data per column.
  - **Model:** cross-validated and held-out accuracy, confusion matrix, ROC curve, feature weights and the model comparison.
  - **Calculators:** EMI with a yearly principal and interest split, affordability, and prepayment savings.

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
| Random forest | 80.9% ± 1.2 | 0.776 |
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

Needs Python 3.10 or newer. Tested on Python 3.10 with scikit-learn 1.3, and on Python 3.13 with scikit-learn 1.9. Always run `train.py` after installing, so the saved model matches your scikit-learn version.

On Windows, the quickest way is to double-click `run.bat`. The first run creates `.venv`, installs the requirements and trains the model. After that it starts the server and opens the dashboard in your browser. Close its window to stop the server.

Or do the same steps by hand:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
uvicorn app:app --reload
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

Open http://127.0.0.1:8000. The page shows "Scored by the Python model through FastAPI" when it's connected.

You can also open `index.html` directly without the server. It falls back to a copy of the model built into the page, so the dashboard still works offline. `train.py` keeps that copy in sync, and a test checks that both give the same answer.

Run the tests with `python -m pytest -q`.

## Put it online

`render.yaml` deploys the server and dashboard to [Render](https://render.com) on the free plan. In Render, choose **New**, then **Blueprint**, and pick this repository. The build installs the requirements and trains the model, then starts the server. Each push to `main` deploys again. Free services sleep when idle, so the first visit after a quiet spell takes about half a minute.

## API

Interactive docs are at http://127.0.0.1:8000/docs once the server is running.

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | Model name, training time, row count |
| GET | `/api/stats` | Data behind the Insights and Model pages |
| POST | `/api/predict` | Scores one application |
| POST | `/api/what-if` | Approval chance across loan amounts from ₹10k to ₹700k |

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
features.py            feature engineering shared by training and the API
train.py               compares models, evaluates, saves the pipeline and dashboard data
app.py                 FastAPI server
index.html             dashboard
run.bat                one-click setup and start on Windows
render.yaml            deploy settings for Render
model/pipeline.joblib  trained pipeline
model/metrics.json     evaluation results
model/dashboard.json   chart data for the dashboard
tests/test_api.py      API and consistency tests
requirements.txt       dependencies
train.csv              dataset
explore.py             first look at the data
```

## Limitations

- 614 rows is small. Scores shift by a point or two between random splits.
- Credit history dominates. Income barely moves the score, so the model can't tell a stretched applicant from a comfortable one if both have clean credit.
- Loan amounts outside ₹9k to ₹700k are beyond what the model has seen, and the API flags them.
- The repayment the model sees is amount divided by term, without interest. The EMI calculator adds interest.
- This is a lab project, not a lending tool.

## Credits

Frontend designed by Harshit Poonia.
