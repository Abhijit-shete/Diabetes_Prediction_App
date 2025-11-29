import streamlit as st
import pandas as pd
import joblib

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

body {
    background: linear-gradient(120deg, #e0f7fa, #ffffff);
    font-family: 'Poppins', sans-serif;
}

.main-title {
    text-align:center;
    font-size:46px;
    font-weight:700;
    color:#1B5E20;
    text-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}

.stButton>button {
    background: linear-gradient(to right, #4caf50, #2e7d32) !important;
    color: white !important;
    font-size: 20px;
    font-weight:600;
    border-radius:12px;
    padding:12px 25px;
    transition: transform 0.2s;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}

.result-card {
    margin-top:20px;
    padding:25px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:600;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.2);
    transition: transform 0.2s;
}
.result-card:hover {
    transform: scale(1.02);
}

.sidebar .sidebar-content {
    background: linear-gradient(to bottom, #e8f5e9, #ffffff);
    border-radius:15px;
    padding:15px;
}

.expander {
    font-size:16px;
    color:#2e7d32;
}
</style>
""", unsafe_allow_html=True)


# ---------------- App Title ----------------
st.markdown("<h1 class='main-title'>🩺 Diabetes Prediction System</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("📌 Enter Your Health Details")

pregnancies = st.sidebar.number_input("👶 Pregnancies", 0, 20, 0)
glucose = st.sidebar.number_input("🍬 Glucose Level", 0, 200, 100)
bp = st.sidebar.number_input("💓 Blood Pressure", 0, 140, 70)
skin = st.sidebar.number_input("📏 Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("💉 Insulin", 0, 900, 100)
bmi = st.sidebar.number_input("⚖ BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("🧬 Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("🎂 Age", 1, 120, 30)

# ---------------- Load model ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- Prediction ----------------
predict = st.button("🚀 Predict")
if predict:
    data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
                        columns=["Pregnancies","Glucose","BloodPressure","SkinThickness",
                                 "Insulin","BMI","DiabetesPedigreeFunction","Age"])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.markdown(
            "<div class='result-card' style='background: linear-gradient(to right, #ff8a80, #d32f2f); color:white;'>⚠ High Risk of Diabetes!</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='result-card' style='background: linear-gradient(to right, #a5d6a7, #2e7d32); color:white;'>✅ Low Risk — You are Safe!</div>",
            unsafe_allow_html=True)

# ---------------- About / Info ----------------
with st.expander("ℹ️ About this Application"):
    st.write("""
    This app predicts diabetes risk using advanced machine learning.
    
    **Metrics used:** Glucose, Blood Pressure, BMI, Insulin, Age, Pedigree Function  
    **Model:** RandomForest / Logistic Regression  
    **Scaling:** StandardScaler  
    **Accuracy:** ~80-90% depending on dataset
    """)
