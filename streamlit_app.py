
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="AI Fraud Detector", page_icon="🛡️", layout="wide")

# Load models — adjust path if running locally
@st.cache_resource
def load_models():
    d = "models"
    return {
        "logreg": joblib.load(os.path.join(d, "logistic_regression.joblib")),
        "xgboost": joblib.load(os.path.join(d, "xgboost.joblib")),
        "scaler": joblib.load(os.path.join(d, "scaler_amount.pkl")),
    }

m = load_models()

st.title("🛡️ AI Fraud Detector")
st.markdown("Enter a credit-card transaction to check if it is fraudulent.")

# Sample buttons
col_s1, col_s2 = st.columns(2)
use_sample = None
if col_s1.button("Load Sample LEGIT Transaction"):
    use_sample = "legit"
if col_s2.button("Load Sample FRAUD Transaction"):
    use_sample = "fraud"

LEGIT = {"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.34,"V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":-0.34,"V25":0.17,"V26":-0.01,"V27":0.01,"V28":-0.02,"Amount":149.62}
FRAUD = {"V1":-2.31,"V2":1.95,"V3":-1.61,"V4":3.99,"V5":-0.52,"V6":-1.43,"V7":-2.54,"V8":1.39,"V9":-2.77,"V10":-2.77,"V11":3.20,"V12":-2.90,"V13":-0.60,"V14":-4.29,"V15":0.39,"V16":-1.14,"V17":-2.83,"V18":-0.02,"V19":0.42,"V20":0.13,"V21":0.66,"V22":-0.05,"V23":-0.15,"V24":0.07,"V25":-0.38,"V26":-0.23,"V27":-0.07,"V28":-0.06,"Amount":2125.87}
defaults = LEGIT if use_sample == "legit" else FRAUD if use_sample == "fraud" else {f"V{i}":0.0 for i in range(1,29)} | {"Amount":100.0}

with st.form("txn"):
    cols = st.columns(4)
    vals = {}
    for i in range(1, 29):
        vals[f"V{i}"] = cols[(i-1)%4].number_input(f"V{i}", value=defaults.get(f"V{i}",0.0), format="%.4f")
    amount = st.number_input("💰 Amount ($)", value=defaults.get("Amount",100.0), format="%.2f")
    go = st.form_submit_button("🔍 Predict", use_container_width=True)

if go:
    row = {**vals, "Amount": amount}
    df = pd.DataFrame([row])
    lr = float(m["logreg"].predict_proba(df)[:,1][0])
    xg = float(m["xgboost"].predict_proba(df)[:,1][0])
    avg = (lr+xg)/2

    c1, c2 = st.columns(2)
    c1.metric("Logistic Regression", f"{lr:.6f}")
    c2.metric("XGBoost", f"{xg:.6f}")
    st.markdown("---")
    if avg > 0.5:
        st.error(f"### 🚨 FRAUD (ensemble probability: {avg:.6f})")
    else:
        st.success(f"### ✅ LEGIT (ensemble probability: {avg:.6f})")
    st.bar_chart(pd.DataFrame({"Model":["LogReg","XGBoost"],"Probability":[lr,xg]}).set_index("Model"))
