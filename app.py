"""Streamlit dashboard - run with:  streamlit run app.py"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from train import add_features  # noqa: E402  (same feature engineering as training)

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load():
    return joblib.load(ROOT / "models" / "churn_model.joblib")


def predict(row: dict):
    art = load()
    df = add_features(pd.DataFrame([row]))[art["columns"]]
    p = float(art["model"].predict_proba(df)[0, 1])
    return p, p >= art["threshold"]


st.title("📉 Customer Churn Prediction")
st.caption("Enter a customer profile to estimate the probability that they will leave.")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Account")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Tenure (months)", 1, 72, 12)
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    monthly = st.number_input("Monthly charges", 18.0, 120.0, 70.0)
with c2:
    st.subheader("Services")
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    sec = st.selectbox("Online security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
with c3:
    st.subheader("Add-ons & demographics")
    prot = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
    tech = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
    senior = st.selectbox("Senior citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict churn risk", type="primary"):
    row = dict(gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=dependents, tenure=tenure,
               PhoneService=phone, MultipleLines=multi, InternetService=internet, OnlineSecurity=sec,
               OnlineBackup=backup, DeviceProtection=prot, TechSupport=tech, StreamingTV=tv,
               StreamingMovies=movies, Contract=contract, PaperlessBilling=paperless, PaymentMethod=payment,
               MonthlyCharges=monthly, TotalCharges=monthly * tenure)
    prob, flag = predict(row)
    st.metric("Churn probability", f"{prob:.1%}")
    st.progress(min(prob, 1.0))
    if flag:
        st.error("HIGH RISK - recommend a retention action.")
        tips = []
        if contract == "Month-to-month":
            tips.append("Offer a discount for moving to a 1- or 2-year contract.")
        if tech == "No" and internet != "No":
            tips.append("Offer free tech support / online security for 3 months.")
        if payment == "Electronic check":
            tips.append("Incentivise automatic payment (bank transfer or card).")
        if tenure < 12:
            tips.append("Trigger an onboarding call in the first year.")
        for t in tips or ["Assign to the retention team for a personal call."]:
            st.write("•", t)
    else:
        st.success("LOW RISK - customer is likely to stay.")

with st.expander("Model performance"):
    import json
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    st.json({k: m[k] for k in ["best_model", "accuracy", "precision", "recall", "f1", "roc_auc"]})
    st.image(str(ROOT / "outputs" / "figures" / "07_feature_importance.png"))
