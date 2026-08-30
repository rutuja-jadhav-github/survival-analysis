"""
app.py
------
The UI only. All ML logic and model loading now live in api.py
(a separate FastAPI service) — this file just sends HTTP requests to
that API and displays whatever comes back.

This means: if this were rebuilt as a React app, a mobile app, or
called by a completely different team, none of them would need to
know anything about RandomSurvivalForest, joblib, or the model's
feature schema — they'd just call the same two endpoints this file
calls.

You must run BOTH of these, in two separate terminals, for the app
to work:

    Terminal 1:  uvicorn api:app --reload --port 8000
    Terminal 2:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

from model import (
    RAW_FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    LOW_RISK_MAX,
    MEDIUM_RISK_MAX,
)

import os
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Telco Churn Risk — RSF Model", layout="wide")
st.title("📉 Telco Customer Churn — Survival Model")

# Quick check that the API is actually reachable, with a clear message if not —
# this is the single most common thing to go wrong once you split into two processes.
try:
    health = requests.get(f"{API_BASE_URL}/health", timeout=3)
    health.raise_for_status()
except requests.exceptions.ConnectionError:
    st.error(
        f"Can't reach the API at {API_BASE_URL}. Is it running?\n\n"
        f"Open a separate terminal and run:\n\n"
        f"    uvicorn api:app --reload --port 8000"
    )
    st.stop()

mode = st.sidebar.radio("Mode", ["Mode A — Batch analysis", "Mode B — Single customer"])

RISK_COLOR = {"Low": "green", "Medium": "orange", "High": "red"}


# ========================================================================
# MODE A — BATCH ANALYSIS
# ========================================================================
if mode == "Mode A — Batch analysis":
    st.subheader("Find your highest-risk customers for a retention campaign")
    st.caption(
        "Upload a CSV of customers (same columns as the standard Telco churn "
        "dataset). Set your target probability and time window, and get back "
        "the subset of customers to prioritize."
    )

    col1, col2 = st.columns(2)
    with col1:
        horizon = st.number_input("Time window (months)", min_value=1, max_value=72, value=1)
    with col2:
        threshold_pct = st.slider("Minimum churn probability (%)", 0, 100, 90)

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)

        missing = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
        if missing:
            st.error(
                f"Your CSV is missing required columns: {missing}\n\n"
                f"Required columns are: {RAW_FEATURE_COLUMNS}"
            )
            st.stop()

        # NaN isn't valid JSON — swap it for None before sending over HTTP
        payload_customers = df.where(pd.notnull(df), None).to_dict(orient="records")

        with st.spinner("Scoring customers via the API..."):
            resp = requests.post(
                f"{API_BASE_URL}/predict/batch",
                json={
                    "customers": payload_customers,
                    "horizon_months": horizon,
                    "threshold_pct": threshold_pct,
                },
                timeout=60,
            )

        if resp.status_code != 200:
            st.error(f"API error ({resp.status_code}): {resp.json().get('detail')}")
            st.stop()

        data = resp.json()
        for w in data["warnings"]:
            st.warning(w)

        filtered = pd.DataFrame(data["results"])

        st.success(
            f"**{data['filtered_count']}** of {data['total_customers']} customers have "
            f"\u2265{threshold_pct}% probability of churning within {horizon} month(s)."
        )
        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            "⬇️ Download this list as CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"retention_targets_{horizon}mo_{threshold_pct}pct.csv",
            mime="text/csv",
        )


# ========================================================================
# MODE B — SINGLE CUSTOMER
# ========================================================================
else:
    st.subheader("Risk-profile a single customer")
    st.caption(
        "Fill in this customer's details. You'll get a Low / Medium / High "
        "risk classification and a survival curve compared against example "
        "risk profiles."
    )

    with st.form("single_customer_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", CATEGORICAL_COLUMNS["gender"])
            senior = st.selectbox("Senior citizen?", ["No", "Yes"])
            partner = st.selectbox("Has partner?", CATEGORICAL_COLUMNS["Partner"])
            dependents = st.selectbox("Has dependents?", CATEGORICAL_COLUMNS["Dependents"])
            phone_service = st.selectbox("Phone service?", CATEGORICAL_COLUMNS["PhoneService"])
            multiple_lines = st.selectbox("Multiple lines?", CATEGORICAL_COLUMNS["MultipleLines"])

        with c2:
            internet_service = st.selectbox("Internet service", CATEGORICAL_COLUMNS["InternetService"])
            online_security = st.selectbox("Online security", CATEGORICAL_COLUMNS["OnlineSecurity"])
            online_backup = st.selectbox("Online backup", CATEGORICAL_COLUMNS["OnlineBackup"])
            device_protection = st.selectbox("Device protection", CATEGORICAL_COLUMNS["DeviceProtection"])
            tech_support = st.selectbox("Tech support", CATEGORICAL_COLUMNS["TechSupport"])

        with c3:
            streaming_tv = st.selectbox("Streaming TV", CATEGORICAL_COLUMNS["StreamingTV"])
            streaming_movies = st.selectbox("Streaming movies", CATEGORICAL_COLUMNS["StreamingMovies"])
            contract = st.selectbox("Contract type", CATEGORICAL_COLUMNS["Contract"])
            paperless = st.selectbox("Paperless billing?", CATEGORICAL_COLUMNS["PaperlessBilling"])
            payment_method = st.selectbox("Payment method", CATEGORICAL_COLUMNS["PaymentMethod"])
            monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, value=65.0, step=1.0)

        horizon_single = st.slider("Risk horizon (months)", 1, 72, 3)
        submitted = st.form_submit_button("Score this customer")

    if submitted:
        customer_payload = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
        }

        resp = requests.post(
            f"{API_BASE_URL}/predict/single",
            json={"customer": customer_payload, "horizon_months": horizon_single},
            timeout=30,
        )

        if resp.status_code != 200:
            st.error(f"API error ({resp.status_code}): {resp.json().get('detail')}")
            st.stop()

        data = resp.json()
        for w in data["warnings"]:
            st.warning(w)

        prob_pct = data["churn_probability_pct"]
        bucket = data["risk_bucket"]
        color = RISK_COLOR[bucket]

        st.markdown(
            f"### Risk classification: :{color}[{bucket}] "
            f"— {prob_pct:.1f}% probability of churn within {horizon_single} months"
        )
        st.caption(
            f"Cutoffs: Low < {LOW_RISK_MAX*100:.0f}%, "
            f"Medium {LOW_RISK_MAX*100:.0f}\u2013{MEDIUM_RISK_MAX*100:.0f}%, "
            f"High > {MEDIUM_RISK_MAX*100:.0f}%"
        )

        curve = data["survival_curve"]
        t, s = curve["times"], curve["survival_probs"]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(t, s, where="post", label="This customer", color="black", linewidth=3)

        ref_colors = {
            "Example: High risk": "tomato",
            "Example: Medium risk": "orange",
            "Example: Low risk": "steelblue",
        }
        for label, curve_data in data["reference_curves"].items():
            ax.step(
                curve_data["times"], curve_data["survival_probs"], where="post",
                label=label, color=ref_colors.get(label, "gray"),
                linewidth=1.5, linestyle="--", alpha=0.8,
            )

        ax.axvline(horizon_single, linestyle=":", color="gray", alpha=0.6, label=f"{horizon_single}-month horizon")
        ax.set_xlabel("Months")
        ax.set_ylabel("Survival probability (retention)")
        ax.set_title("Predicted survival curve vs. example risk profiles")
        ax.set_ylim(0, 1.02)
        ax.legend()
        st.pyplot(fig)

        st.caption(
            "Note: this is the model's predicted survival curve for this specific "
            "customer, not an empirical Kaplan-Meier curve (KM curves are computed "
            "from a group, not one individual). The dashed lines are example "
            "archetype profiles for visual comparison, not real customers."
        )