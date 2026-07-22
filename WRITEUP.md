# Business Write-Up — Customer Churn Prediction

## Problem

Customer churn — a customer canceling their service — is one of the most direct threats to
recurring revenue in subscription-based businesses like telecoms and banks. Acquiring a new
customer is typically far more expensive than retaining an existing one, so being able to flag
at-risk customers *before* they leave allows a business to intervene early (discounts, outreach,
service fixes) rather than losing the customer entirely.

This project predicts whether a customer will churn, based on their account details, contract
type, billing method, and service usage — giving a business a ranked list of at-risk customers
to act on.

## Data & Approach

- **Dataset:** Telco Customer Churn (Kaggle), ~7,000 customers, 20 features covering
  demographics, account tenure, contract type, billing, and subscribed services.
- **Class balance:** ~74% stayed, ~26% churned — an imbalanced target, handled explicitly
  rather than ignored.
- **Models compared:**
  1. Logistic Regression — a simple, interpretable baseline
  2. XGBoost — a gradient-boosted tree model, tuned with Optuna (50-trial search,
     5-fold cross-validation)
- **Validation:** an 80/20 stratified train/test split; the test set was touched exactly once,
  for final evaluation only, after all tuning was complete.

## Results

| Metric | Logistic Regression | XGBoost (untuned) | XGBoost (tuned) |
|---|---|---|---|
| ROC-AUC | **0.851** | 0.834 | 0.835 |
| Precision (churn) | 0.67 | 0.56 | 0.51 |
| Recall (churn) | 0.55 | 0.69 | **0.79** |
| F1-score (churn) | 0.60 | 0.62 | 0.62 |
| Accuracy | 0.80 | 0.77 | 0.74 |

No single model wins outright, and that is a meaningful finding in itself, not an inconclusive
one. Logistic regression ranks customers by churn risk slightly more reliably overall
(ROC-AUC). Tuned XGBoost catches substantially more of the customers who actually churn
(recall), at the cost of more false alarms among loyal customers (precision).

## Business Interpretation

At the tuned XGBoost model's default threshold, on the 1,407-customer test set:

- **302 of 380 actual churners were correctly identified** (79% recall)
- **78 churners were missed**
- **285 loyal customers were incorrectly flagged as at-risk** (false alarms)

**Recommendation: prioritize recall over precision for this use case.** A missed churner is
lost recurring revenue that must be replaced by acquiring a new customer, which is typically
more expensive than the cost of a retention offer sent to a loyal customer who wasn't actually
going to leave. On that basis, the tuned XGBoost model is the recommended model — it catches
79% of at-risk customers instead of 55%, a substantial improvement worth the added false-alarm
cost. This is a business-priority judgment, not a purely statistical one; a business with a
very low retention-offer budget might reasonably choose differently.

## Key Drivers (via SHAP)

- **Contract length** is the strongest predictor — customers on two-year contracts are
  markedly less likely to churn than those on month-to-month contracts.
- **Tenure** — newer customers churn more than long-tenured ones, consistent with the
  contract-length finding.
- **Fiber optic internet** and **electronic check payment method** were both associated with
  higher churn risk — plausibly linked to price sensitivity or service satisfaction, worth
  further investigation with the business rather than assumed.

These patterns were first observed during exploratory analysis and later confirmed
quantitatively through SHAP values on the final model, giving confidence they reflect a real
signal rather than noise.

## Limitations

- The dataset reflects a single snapshot in time; customer behavior and churn drivers may shift,
  and the model would need periodic retraining to stay reliable.
- The model has not been validated against a different customer base or market; results may not
  generalize directly outside this dataset's population.
- The recall/precision tradeoff was evaluated at the model's default threshold; a business
  could further tune this threshold to match its specific cost tolerance for false alarms versus
  missed churners.
