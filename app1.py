import streamlit as st
import pickle
import pdfplumber
import pandas as pd
import numpy as np
import tempfile
import os

# ----------------- LOAD MODEL + SCALER -----------------
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

if os.path.exists(MODEL_PATH):
    model = pickle.load(open(MODEL_PATH, "rb"))
else:
    model = None

if os.path.exists(SCALER_PATH):
    scaler = pickle.load(open(SCALER_PATH, "rb"))
else:
    scaler = None

# ------------- Ordered Features (VERY IMPORTANT) -------------
FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# ------------- Risk Labels ----------------
def risk_category(prob):
    if prob >= 0.75:
        return "High Risk 🔴"
    elif prob >= 0.50:
        return "Moderate Risk 🟠"
    else:
        return "Low Risk 🟢"

# ------------- Extract Numbers From PDF ----------------
def extract_numbers(text):
    nums = []
    for token in text.replace("\n", " ").split(" "):
        token = token.strip().replace(",", "")
        if token.replace(".", "", 1).isdigit():
            nums.append(float(token))
    return nums

# ----------------- UI -----------------
st.title("🩺 Diabetes Report Prediction System")
st.write("Upload your medical report PDF and get instant diabetes risk prediction.")

uploaded = st.file_uploader("Upload PDF Report", type=["pdf"])

if uploaded:
    st.success("PDF uploaded ✔")

    with pdfplumber.open(uploaded) as pdf:
        text = ""
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text += txt + "\n"

    values = extract_numbers(text)

    if len(values) < len(FEATURE_ORDER):
        st.warning(f"⚠ Extracted only {len(values)} values. Required: {len(FEATURE_ORDER)}")
        st.write(values)
        st.stop()

    feat_vec = values[:len(FEATURE_ORDER)]
    df = pd.DataFrame([feat_vec], columns=FEATURE_ORDER)

    st.subheader("📌 Extracted Values")
    st.write(df)

    if model is None or scaler is None:
        st.error("Model or Scaler missing ❌")
    else:
        scaled = scaler.transform(df)
        prob = model.predict_proba(scaled)[0][1]
        pred_label = risk_category(prob)

        st.subheader("📊 Prediction Result")
        st.metric("Risk Level", pred_label)
        st.write(f"Probability: **{prob:.2f}**")

        # Advice
        if "High" in pred_label:
            st.error("❗High chance of diabetes — Please consult a doctor.")
        elif "Moderate" in pred_label:
            st.warning("⚠ Medium risk — Monitor health & lifestyle.")
        else:
            st.success("✔ Low Risk — Keep maintaining healthy lifestyle!")

        # Save History
        df["Result"] = pred_label
        df["Probability"] = prob

        if os.path.exists("history.csv"):
            df.to_csv("history.csv", mode="a", header=False, index=False)
        else:
            df.to_csv("history.csv", index=False)

        st.success("📁 Prediction saved to history file.")

else:
    st.info("Upload a valid PDF to start.")
