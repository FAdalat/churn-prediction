# Credit Risk / Churn Prediction

A binary classification project predicting whether a telecom customer will churn (cancel their
service), based on account and usage data.

**Status:** Complete. Logistic Regression baseline and Optuna-tuned XGBoost compared honestly
on a held-out test set, with SHAP explainability on the final model. See
[WRITEUP.md](WRITEUP.md) for the full business write-up.

## Project Overview

This project predicts customer churn using the
[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(~7,000 customers). The goal was not just to build a model that predicts churn, but to do it
with the rigor a real business use case demands: proper validation, honest evaluation on
imbalanced data, and explainable predictions.

## Dataset

- **Source:** Kaggle — Telco Customer Churn
- **Size:** ~7,000 rows, 21 original columns
- **Target:** `Churn` (Yes/No) — encoded to 1/0
- **Class balance:** ~74% stayed, ~26% churned (imbalanced)

## Results

| Metric | Logistic Regression | XGBoost (untuned) | XGBoost (tuned) |
|---|---|---|---|
| ROC-AUC | 0.851 | 0.834 | 0.835 |
| Precision (churn) | 0.67 | 0.56 | 0.51 |
| Recall (churn) | 0.55 | 0.69 | **0.79** |
| F1-score (churn) | 0.60 | 0.62 | 0.62 |

**Recommended model: tuned XGBoost** — for this use case, catching more actual churners
(higher recall) outweighs the cost of additional false alarms, since a missed churner is lost
recurring revenue while a false alarm only costs a retention offer to a loyal customer. Full
reasoning and the confusion matrix breakdown are in [WRITEUP.md](WRITEUP.md).

### Key findings (SHAP explainability)
- **Contract length** is the strongest churn predictor — long-term contracts reduce churn risk
- **Tenure** — newer customers churn more than long-tenured ones
- **Fiber optic internet** and **electronic check payment** are both associated with higher
  churn risk

## Tech Stack

- **Data handling:** pandas, numpy
- **Modeling:** scikit-learn, XGBoost
- **Tuning:** Optuna
- **Explainability:** SHAP
- **Visualization:** matplotlib, seaborn

## How to Reproduce

```bash
pip install -r requirements.txt
jupyter lab
```

Open `main.ipynb` and run cells in order — each checkpoint builds on the
previous one. Charts are saved to `outputs/charts/` as they're generated.

## Repository Structure

```
churn-prediction/
├── outputs/
│   └── charts/              # EDA and SHAP visualizations
├── telco_churn.csv
├── main.ipynb
├── requirements.txt
├── README.md
├── LICENSE
└── WRITEUP.md                # full business write-up
```

## Project Workflow

1. Load and inspect the raw data
2. Exploratory analysis — class balance, tenure/charges/contract vs. churn
3. Clean and encode — fix `TotalCharges`, one-hot encode categoricals
4. Stratified 80/20 train/test split
5. Baseline model — scaled Logistic Regression
6. Main model — XGBoost with class imbalance handling
7. Hyperparameter tuning — Optuna, 50 trials, 5-fold cross-validation
8. Final evaluation — held-out test set, touched exactly once
9. Explainability — SHAP global and individual-prediction analysis
10. Business write-up — see [WRITEUP.md](WRITEUP.md)
