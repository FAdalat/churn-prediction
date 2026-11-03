# Telco Customer Churn Prediction

A machine-learning project that predicts whether a telecommunications
customer is likely to leave the service (**churn**) using customer
account, contract, billing, and service information.

The project compares Logistic Regression with baseline and Optuna-tuned
XGBoost models. It uses a stratified hold-out test set and evaluates
performance with metrics suited to an imbalanced classification problem.

## Project goals

-   Explore customer churn patterns in a real-world-style telecom
    dataset.
-   Build a reproducible classification workflow.
-   Compare a linear baseline with a gradient-boosted tree model.
-   Tune XGBoost hyperparameters using cross-validation on the training
    data.
-   Evaluate the final candidates on a separate hold-out test set.
-   Interpret model performance in terms of a practical
    customer-retention use case.

## Dataset

-   **Dataset:** Telco Customer Churn
-   **Original source:** [Kaggle --- Telco Customer
    Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
-   **Target:** `Churn` (`Yes` = 1, `No` = 0)
-   **Task:** Binary classification

The target is imbalanced: more customers stayed than churned. For that
reason, accuracy is not treated as the only measure of model quality.
Precision, recall, F1-score, ROC-AUC, and average precision are also
considered.

The dataset is provided in `telco_churn.csv`. Please retain the
dataset's original attribution and review its source terms before
redistributing it.

## Models and workflow

1.  Load the dataset and inspect its columns, data types, missing
    values, and target distribution.
2.  Explore relationships between churn and selected customer
    attributes, including tenure, monthly charges, and contract type.
3.  Split the data into training and test sets using stratification.
4.  Fit preprocessing within model pipelines to reduce data leakage.
5.  Train a Logistic Regression baseline.
6.  Train a baseline XGBoost classifier.
7.  Tune XGBoost with Optuna using stratified cross-validation on the
    training data.
8.  Compare the selected models on the hold-out test set.
9.  Review confusion matrices, ROC curves, and precision-recall curves.
10. Discuss the business implications and limitations of the results.

The hold-out test set is reserved for final evaluation; model selection
and hyperparameter tuning should use training data only.

## Results

The following values are based on the confusion matrices and evaluation
plots generated in the executed notebook. Precision, recall, F1-score,
and accuracy are calculated from the displayed confusion matrices.
ROC-AUC and average precision (AP) are read from the plotted legends and
rounded to two decimal places.

  Metric                     Logistic Regression   XGBoost baseline   XGBoost tuned
  ------------------------ --------------------- ------------------ ---------------
  Accuracy                                 0.806              0.755           0.743
  Churn precision                          0.657              0.526           0.510
  Churn recall                             0.559              0.773       **0.805**
  Churn F1-score                           0.604          **0.626**           0.626
  ROC-AUC                                   0.84               0.84        **0.85**
  Average precision (AP)                    0.63               0.65        **0.66**

### Confusion matrix counts

Rows represent the true class; columns represent the predicted class.
Class `0` means **Stayed** and class `1` means **Churned**.

  ------------------------------------------------------------------------
  Model         True stayed /  True stayed / True churned / True churned /
                    predicted      predicted      predicted      predicted
                       stayed        churned         stayed        churned
  ------------ -------------- -------------- -------------- --------------
  Logistic                926            109            165            209
  Regression                                                

  XGBoost                 775            260             85            289
  baseline                                                  

  XGBoost                 746            289             73            301
  tuned                                                     
  ------------------------------------------------------------------------

### Interpretation

The tuned XGBoost model has the highest churn recall (about 80.5%),
meaning it identifies a larger share of customers who actually churned.
It also has the highest ROC-AUC and average precision in these results.

That improvement comes with a cost: its churn precision is lower, so
more customers who would have stayed are incorrectly flagged as
potential churners. Logistic Regression has the highest accuracy and
churn precision, but misses more actual churners.

There is no universally best model. If the business priority is to
identify as many potential churners as possible, tuned XGBoost is a
reasonable candidate. If retention outreach is expensive and false
alarms must be limited, Logistic Regression or a different decision
threshold may be preferable. The threshold should ultimately be chosen
using business costs and validation data, not by looking for the best
result on the test set.

## Visual outputs

When the notebook is run, it saves charts in `outputs/charts/`. The main
figures are:

-   `01_eda_overview.png` --- target distribution and exploratory
    comparisons.
-   `02_model_curves.png` --- ROC and precision-recall curves.
-   `03_confusion_matrices.png` --- confusion matrices for the three
    models.

## Technology stack

-   Python
-   pandas and NumPy
-   scikit-learn
-   XGBoost
-   Optuna
-   matplotlib and seaborn
-   Jupyter Notebook / JupyterLab
-   SHAP (optional explainability section; see the note below)

## Setup

Python **3.11 (64-bit)** is recommended for this project because it is
compatible with the pinned scientific Python package versions used in
the original environment.

### Windows PowerShell

Open the project folder in VS Code, then run:

``` powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Install the packages listed in `requirements.txt`:

``` powershell
python -m pip install -r requirements.txt
```

If you have adjusted package versions to match the wheels available for
your Python installation, keep those changes documented in
`requirements.txt` so the environment can be reproduced.

Register the environment as a Jupyter kernel:

``` powershell
python -m pip install ipykernel
python -m ipykernel install --user --name churn-prediction --display-name "Python 3.11 - Churn Prediction"
```

In VS Code:

1.  Open `main.ipynb`.
2.  Select the **Python 3.11 - Churn Prediction** kernel (or the
    interpreter inside this project's `.venv`).
3.  Run the notebook cells from top to bottom.
4.  Save the notebook after execution.

To verify the notebook is using the intended environment, run this in a
notebook cell:

``` python
import sys
print(sys.executable)
```

The printed path should end with `.venv\\Scripts\\python.exe`.

## SHAP / Windows security note

The SHAP explainability section depends on Numba, which includes
compiled components. On some Windows systems, Smart App Control or
another application-control policy may block a Numba DLL even when SHAP
and Numba are installed correctly.

If this happens, do not disable Windows security protections just to run
this project. You can run the preprocessing, model training, Optuna
tuning, and standard evaluation sections while leaving the SHAP section
unexecuted. Record that explainability was not run in that environment,
and use an approved environment or administrator-supported process if
SHAP is required.

## Repository structure

``` text
churn-prediction/
├── outputs/
│   └── charts/                 # Generated EDA and model evaluation charts
├── telco_churn.csv             # Dataset
├── main.ipynb                  # Main analysis and modeling notebook
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview and instructions
├── WRITEUP.md                  # Business interpretation and limitations
├── LICENSE
└── .gitignore
```

## Limitations

-   The dataset is a snapshot; churn patterns can change over time.
-   Results may not generalize to other telecom providers, regions, or
    customer populations.
-   Predictive associations do not establish that a feature causes
    churn.
-   The default classification threshold may not reflect the costs of
    false positives and false negatives for a real business.
-   A production deployment would require monitoring, periodic
    validation, retraining where appropriate, and checks for data
    quality and model drift.
-   The notebook's SHAP analysis may need to be run in an environment
    where Numba's compiled components are permitted.

## Further reading

See [`WRITEUP.md`](WRITEUP.md) for the business interpretation, model
trade-offs, and project limitations.
