import streamlit as st
import pandas as pd
import joblib

# -------------- Custom CSS Styling ----------------
st.markdown("""
<style>
    body {
        background: linear-gradient(120deg, #e3f2fd, #ffffff);
    }
    .main-title {
        text-align:center;
        color:#2E8B57;
        font-size:42px;
        font-weight: bold;
        text-shadow: 1px 1px 10px rgba(0,0,0,0.2);
    }
    .predict-btn button {
        background-color:#4CAF50 !important;
        color:white !important;
        border-radius:12px;
        padding: 12px 25px;
        font-size:18px;
        font-weight:bold;
        transition:0.3s;
    }
    .predict-btn button:hover {
        background-color:#2E7D32 !important;
        scale: 1.05;
    }
    .result-card {
        padding:20px;
        border-radius:12px;
        font-size:22px;
        font-weight:bold;
        text-align:center;
        box-shadow:0px 4px 15px rgba(0,0,0,0.15);
        margin-top:20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- App Title --------------------
st.markdown("<h1 class='main-title'>🩺 Diabetes Prediction System</h1>", unsafe_allow_html=True)
st.markdown("---")


# --------------- Sidebar Input -------------------
st.sidebar.header("📌 Enter Health Details")

pregnancies = st.sidebar.number_input("👶 Pregnancies", 0, 20, 0)
glucose = st.sidebar.number_input("🍬 Glucose Level", 0, 200, 100)
bp = st.sidebar.number_input("💓 Blood Pressure", 0, 140, 70)
skin = st.sidebar.number_input("📏 Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("💉 Insulin", 0, 900, 100)
bmi = st.sidebar.number_input("⚖ BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("🧬 Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("🎂 Age", 1, 120, 30)


# ---------------------- Load Model -----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# ---------------------- Prediction Logic -----------------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    predict = st.button("🚀 Predict", key="predict_btn")
    
if predict:
    data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
                        columns=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"])
    
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.markdown("<div class='result-card' style='background-color:#ffcccc; color:#b30000;'>⚠ High Risk of Diabetes!</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-card' style='background-color:#d9fdd3; color:#1b5e20;'>✅ Low Risk — You are Safe!</div>", unsafe_allow_html=True)


# ---------------------- Additional Info -----------------
with st.expander("ℹ️ About this Application"):
    st.write("""
    This diabetes prediction system uses machine learning and medical metrics 
    such as glucose level, insulin, BMI and age to estimate your risk level.

    ✔ Model: RandomForest / Logistic Regression  
    ✔ Scaling: StandardScaler  
    ✔ Accuracy: ~80-90% (Varies by dataset)
    """)
