"""
api.py
------
A small FastAPI service that wraps model.py's logic behind two REST
endpoints. This is a documented HTTP contract that any client (Streamlit, a mobile app, another team's
service, curl) could call — not just this specific Streamlit script.

Run with:
    uvicorn api:app --reload --port 8000

Then open http://localhost:8000/docs for interactive, auto-generated
API documentation (this comes free from FastAPI + the Pydantic models
below — nothing extra to write).
"""

from typing import List, Dict, Any, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

import model as m

app = FastAPI(
    title="Telco Churn Survival API",
    description="Serves churn-risk predictions from a trained Random Survival Forest.",
    version="1.0.0",
)

# Load the model once, when the API process starts — not on every request.
_model = m.load_model()


# ------------------------------------------------------------------
# Request / response schemas (Pydantic)
# These are the "contract": FastAPI validates every incoming request
# against these before your code ever runs, and rejects anything that
# doesn't match with a clear error message.
# ------------------------------------------------------------------
class CustomerFeatures(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["No", "Yes"]
    Dependents: Literal["No", "Yes"]
    PhoneService: Literal["No", "Yes"]
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "No internet service", "Yes"]
    OnlineBackup: Literal["No", "No internet service", "Yes"]
    DeviceProtection: Literal["No", "No internet service", "Yes"]
    TechSupport: Literal["No", "No internet service", "Yes"]
    StreamingTV: Literal["No", "No internet service", "Yes"]
    StreamingMovies: Literal["No", "No internet service", "Yes"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["No", "Yes"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(ge=0)


class SinglePredictionRequest(BaseModel):
    customer: CustomerFeatures
    horizon_months: float = Field(gt=0, le=72)
    current_tenure_months: float = Field(
        default=0, ge=0, le=72,
        description="How many months this customer has already been active. "
                     "0 means treat as a brand-new customer.",
    )


class SurvivalCurve(BaseModel):
    times: List[float]
    survival_probs: List[float]


class SinglePredictionResponse(BaseModel):
    churn_probability_pct: float
    risk_bucket: str
    survival_curve: SurvivalCurve
    reference_curves: Dict[str, SurvivalCurve]
    warnings: List[str]


class BatchPredictionRequest(BaseModel):
    customers: List[Dict[str, Any]]  # rows from the uploaded CSV, as dicts
    horizon_months: float = Field(gt=0, le=72)
    threshold_pct: float = Field(default=0, ge=0, le=100)


class BatchPredictionResponse(BaseModel):
    total_customers: int
    filtered_count: int
    results: List[Dict[str, Any]]
    warnings: List[str]


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/health")
def health():
    """Basic liveness check — useful once this is deployed somewhere."""
    return {"status": "ok", "model_features": int(_model.n_features_in_)}


@app.post("/predict/single", response_model=SinglePredictionResponse)
def predict_single(request: SinglePredictionRequest):
    row = pd.DataFrame([request.customer.model_dump()])
    X_encoded, warnings = m.encode_features(row, _model)

    prob = m.churn_probability(
        _model, X_encoded, request.horizon_months,
        current_tenure=[request.current_tenure_months],
    )[0]
    bucket = m.risk_bucket(prob)
    t, s = m.get_survival_curve(_model, X_encoded)

    reference_curves = {
        label: SurvivalCurve(times=list(x), survival_probs=list(y))
        for label, (x, y) in m.build_reference_curves(_model).items()
    }

    return SinglePredictionResponse(
        churn_probability_pct=round(float(prob) * 100, 2),
        risk_bucket=bucket,
        survival_curve=SurvivalCurve(times=list(t), survival_probs=list(s)),
        reference_curves=reference_curves,
        warnings=warnings,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    if not request.customers:
        raise HTTPException(status_code=400, detail="No customers provided.")

    df = pd.DataFrame(request.customers)

    missing = m.validate_raw_columns(df)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Uploaded data is missing required columns: {missing}",
        )

    X_encoded, warnings = m.encode_features(df, _model)

    # If the uploaded CSV includes a 'tenure' column (common, since it's a
    # standard Telco column even though it's not a model feature), use each
    # customer's actual elapsed tenure to compute a correctly conditioned
    # churn probability instead of treating everyone as brand new.
    if "tenure" in df.columns:
        current_tenure = pd.to_numeric(df["tenure"], errors="coerce")
        n_missing = current_tenure.isna().sum()
        if n_missing > 0:
            warnings.append(
                f"{n_missing} row(s) had a missing/invalid 'tenure' value and "
                f"were treated as brand-new customers (tenure=0)."
            )
        current_tenure = current_tenure.fillna(0.0).values
    else:
        current_tenure = None
        warnings.append(
            "No 'tenure' column found in the uploaded data \u2014 every customer "
            "was treated as brand new (tenure=0). If these are existing active "
            "customers, include their current tenure for accurate probabilities."
        )

    probs = m.churn_probability(_model, X_encoded, request.horizon_months, current_tenure=current_tenure)

    results_df = df.copy()
    results_df.insert(0, "churn_probability_pct", (probs * 100).round(1))
    results_df.insert(1, "risk_bucket", [m.risk_bucket(p) for p in probs])

    filtered = results_df[
        results_df["churn_probability_pct"] >= request.threshold_pct
    ].sort_values("churn_probability_pct", ascending=False)

    # Replace NaN with None so it's valid JSON
    filtered = filtered.where(pd.notnull(filtered), None)

    return BatchPredictionResponse(
        total_customers=len(results_df),
        filtered_count=len(filtered),
        results=filtered.to_dict(orient="records"),
        warnings=warnings,
    )
