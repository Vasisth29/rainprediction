# ---------- imports ----------
import streamlit as st
import numpy as np, pandas as pd, joblib

# ---------- load artefacts once ----------
@st.cache_resource
def load_artifacts():
    rf  = joblib.load("rainfall_rf_model.pkl")
    lr  = joblib.load("rainfall_lr_model.pkl")
    leS = joblib.load("le_subdivision.pkl")
    leM = joblib.load("le_month.pkl")
    return rf, lr, leS, leM

rf_model, lr_model, le_sub, le_month = load_artifacts()

# ---------- helper ----------
def predict_rainfall(subdivision, year, month):
    sub_enc   = le_sub.transform([subdivision])[0]
    month_enc = le_month.transform([month])[0]
    X_new     = np.array([[sub_enc, year, month_enc]])

    if 1901 <= year <= 2017:
        return rf_model.predict(X_new)[0]
    else:
        return lr_model.predict(X_new)[0]

# ---------- UI ----------
st.title("🌧️ India Rainfall Prediction")

st.markdown(
"""
Choose a **sub‑division**, **month**, and **year**.<br>
Years **1901‑2017** use a Random Forest, while earlier/later years use Linear Regression
for extrapolation.
""",
unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    subdivision = st.selectbox("Subdivision", list(le_sub.classes_))
    month       = st.selectbox("Month", list(le_month.classes_))
with col2:
    year = st.number_input("Year", min_value=1800, max_value=2100, value=2025, step=1)

if st.button("Predict"):
    rainfall = predict_rainfall(subdivision, int(year), month)
    st.metric("Predicted rainfall (mm)", f"{rainfall:,.2f}")

    # simple bar chart
    st.bar_chart(
        pd.DataFrame({"Rainfall (mm)": [rainfall]}, index=[f"{month}-{year}"])
    )
