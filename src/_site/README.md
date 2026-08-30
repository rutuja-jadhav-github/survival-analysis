# Telco Churn Risk App

A Streamlit app built around your trained Random Survival Forest model
(`rsf_final_model.pkl`). Two modes:

- **Mode A — Batch analysis**: upload a CSV of customers, set a churn
  probability threshold and a time window, get back the customers to
  target for a retention campaign.
- **Mode B — Single customer**: fill in one customer's details, get a
  Low/Medium/High risk label and a survival curve plotted against
  example risk profiles.

## Files

```
telco_app/  (this is your existing `src` folder)
├── app.py                 <- the UI (Streamlit)
├── model.py               <- all the ML logic (loading model, encoding, prediction)
├── rsf_final_model.pkl    <- your trained model (already here)
└── requirements.txt
```

All files live flat in the same folder — no subfolder needed for the model.

## How to run this on your own machine

1. Open a terminal in this folder (`telco_app/`).

2. Create a virtual environment (keeps this project's packages separate
   from everything else on your computer) and activate it:

   ```
   python -m venv venv
   source venv/bin/activate        # Mac/Linux
   venv\Scripts\activate           # Windows
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

4. Run the app:

   ```
   streamlit run app.py
   ```

5. It will open automatically in your browser (usually at
   `http://localhost:8501`). If not, copy the URL printed in the
   terminal into your browser.

## For Mode A: what your CSV needs

Your CSV must contain these columns (same names as the standard IBM
Telco churn dataset — extra columns like `customerID`, `tenure`, or
`TotalCharges` are fine and will just be carried through to the output,
not used by the model):

```
gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
PaymentMethod, MonthlyCharges
```

## A note on how predictions work

Your model doesn't predict "will this customer churn, yes/no" directly.
It predicts a full **survival curve** for each customer — the
probability they're still a customer at every future month. To get
"probability of churn within M months," the app reads that curve at
month M and reports `1 - survival probability`. This is why you can ask
for any time window (1 month, 3 months, 12 months) without retraining
anything.

## What this is (and isn't) yet

This is what's called **Layer 1** of a deployment pipeline: model + UI
in one Streamlit app, running on your own machine. It's a complete,
legitimate, demo-able tool as-is. If you later want to:

- show "API design" skills for job applications → we split the model
  logic into a small FastAPI service that Streamlit calls
- make it runnable identically on any machine → add a Dockerfile
- give other people a link to use it without installing anything →
  deploy it to a free host (Render, Railway, Hugging Face Spaces)

None of that is required for this to work today — those are optional
next steps, not missing pieces.
