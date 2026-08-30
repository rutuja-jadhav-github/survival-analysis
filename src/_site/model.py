"""
model.py
--------
All the "ML" logic lives here, separate from any UI or API framework.
This file has NO dependency on Streamlit or FastAPI — it's pure Python
+ pandas/numpy/joblib. That's what makes it reusable: api.py (FastAPI)
calls these same functions, and app.py (Streamlit) used to call them
directly before we added the API layer.

What this file does:
  1. Loads your trained RSF model (rsf_final_model.pkl)
  2. Encodes raw customer data into the exact 28 columns your model
     expects (matching what pd.get_dummies(drop_first=True) produced
     during training, but done manually so it's safe for a single
     customer or a small batch, not just a full dataset)
  3. Turns the RSF's predicted survival function into:
       - probability of churn within a given horizon (months)
       - a Low / Medium / High risk label
       - the full survival curve for plotting
"""

import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "rsf_final_model.pkl"

# ----------------------------------------------------------------------
# Feature schema — derived directly from your notebook's preprocessing.
# These are the RAW columns a user's CSV (or manual form) must provide.
# ----------------------------------------------------------------------
NUMERIC_COLUMNS = ["SeniorCitizen", "MonthlyCharges"]

CATEGORICAL_COLUMNS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

RAW_FEATURE_COLUMNS = NUMERIC_COLUMNS + list(CATEGORICAL_COLUMNS.keys())

# ----------------------------------------------------------------------
# Risk bucket cutoffs — probability of churn within a 12-month horizon.
#
# These are a BUSINESS POLICY choice, not a statistical one — how much
# churn risk counts as "high" depends on the company's churn tolerance,
# not on the model. Edit these two numbers freely.
#
# Calibrated against the model's actual 12-month churn probability
# distribution across all real Telco customers:
#   Low (<10%):     47.1% of customers
#   Medium (10-30%): 30.8% of customers
#   High (>=30%):    22.1% of customers
#
# NOTE: churn probability necessarily grows with the horizon (more time
# = more chances to churn) — these cutoffs were calibrated specifically
# for a 12-month horizon. If you change the horizon significantly (e.g.
# to 36 or 72 months), these same absolute percentages will classify a
# very different share of customers as high-risk — worth re-checking the
# distribution at that horizon before trusting the buckets.
# ----------------------------------------------------------------------
LOW_RISK_MAX = 0.10
MEDIUM_RISK_MAX = 0.30


def load_model():
    """Loads the trained RandomSurvivalForest from disk."""
    return joblib.load(MODEL_PATH)


def validate_raw_columns(df: pd.DataFrame):
    """Returns a list of missing required columns, if any."""
    return [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]


def encode_features(df_raw: pd.DataFrame, model):
    """
    Turns raw customer columns into the exact 28 one-hot columns
    the model was trained on (model.feature_names_in_), in the
    exact same order. Any category not seen during training is
    simply treated as the "baseline" (all dummy columns = 0) for
    that feature rather than crashing.

    Returns (X_encoded, warnings) — warnings is a list of strings
    describing any unrecognized category values encountered, so the
    caller (API or UI) can decide how to display them.
    """
    out = pd.DataFrame(index=df_raw.index)
    warnings = []

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(df_raw[col], errors="coerce")

    for col, categories in CATEGORICAL_COLUMNS.items():
        sorted_cats = sorted(categories)
        baseline = sorted_cats[0]
        raw_values = df_raw[col].astype(str)

        unseen = set(raw_values.unique()) - set(categories)
        if unseen:
            warnings.append(
                f"Column '{col}' contains value(s) not seen during training: "
                f"{sorted(unseen)}. These rows will be treated as the "
                f"baseline category ('{baseline}')."
            )

        for cat in sorted_cats[1:]:  # drop_first behaviour, matches training
            out[f"{col}_{cat}"] = (raw_values == cat).astype(int)

    # Force exact column order the model expects; fill anything missing with 0
    out = out.reindex(columns=list(model.feature_names_in_), fill_value=0)
    return out, warnings


def churn_probability(model, X_encoded: pd.DataFrame, horizon_months: float, current_tenure=None) -> np.ndarray:
    """
    P(churn within the next horizon_months | already survived current_tenure months),
    computed as 1 - S(t0+M)/S(t0) for each row.

    current_tenure: array-like of each customer's elapsed tenure so far (t0).
    If None, defaults to 0 for every row — i.e. treats every customer as brand
    new, which reduces this exact formula to the plain 1 - S(M) case, since
    S(0) = 1. This is why there's only one function: t0=0 isn't a special
    case that needs separate code, it falls out of the same formula.
    """
    surv_funcs = model.predict_survival_function(X_encoded, return_array=False)
    n = len(surv_funcs)

    if current_tenure is None:
        current_tenure = np.zeros(n)
    else:
        current_tenure = np.asarray(current_tenure, dtype=float)

    min_t = model.unique_times_.min()
    max_t = model.unique_times_.max()

    probs = []
    for fn, t0 in zip(surv_funcs, current_tenure):
        t0_clipped = min(max(t0, 0.0), max_t)
        t1_clipped = min(t0_clipped + horizon_months, max_t)

        s0 = fn(t0_clipped)
        s1 = fn(t1_clipped)

        if s0 <= 1e-9:
            # The model has essentially no confidence this customer survives
            # to t0 at all (extremely rare in practice) — nothing meaningful
            # to condition on, so treat as certain churn rather than divide
            # by ~zero.
            probs.append(1.0)
        else:
            probs.append(1 - (s1 / s0))

    return np.array(probs)


def get_survival_curve(model, X_encoded_single_row: pd.DataFrame):
    """Returns (time_points, survival_probs) for one customer."""
    fn = model.predict_survival_function(X_encoded_single_row, return_array=False)[0]
    return fn.x, fn.y


def risk_bucket(prob: float) -> str:
    if prob < LOW_RISK_MAX:
        return "Low"
    elif prob < MEDIUM_RISK_MAX:
        return "Medium"
    return "High"


# ----------------------------------------------------------------------
# Reference example profiles for Mode B comparison plot.
# These are illustrative "archetype" customers (not real data) so a
# manager can visually compare an entered customer against clearly
# different risk levels on the same chart.
# ----------------------------------------------------------------------
REFERENCE_PROFILES = {
    "Example: Low risk": {  # actual 12-month churn prob: 1.9%
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Two year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 20.75,
    },
    "Example: Medium risk": {  # actual 12-month churn prob: 19.9%
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 69.4,
    },
    "Example: High risk": {  # actual 12-month churn prob: 41.7%
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 74.4,
    },
}


def build_reference_curves(model):
    """Encodes each reference profile and returns {label: (x, y)} survival curves."""
    curves = {}
    for label, profile in REFERENCE_PROFILES.items():
        row = pd.DataFrame([profile])
        X_encoded, _ = encode_features(row, model)
        x, y = get_survival_curve(model, X_encoded)
        curves[label] = (x, y)
    return curves
