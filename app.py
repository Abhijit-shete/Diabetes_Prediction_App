import streamlit as st
import pandas as pd
import joblib


st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🩺 Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")


st.sidebar.header("Enter Your Details")
pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 0)
glucose = st.sidebar.number_input("Glucose", 0, 200, 100)
bp = st.sidebar.number_input("Blood Pressure", 0, 140, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 100)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("Age", 1, 120, 30)

# ------------------------------
# Load model + scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# ------------------------------
# Predict button
if st.button("Predict"):
    # Prepare input dataframe
    data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
                        columns=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"])
    
    # Scale if using scaler
    data_scaled = scaler.transform(data)
    
    # Prediction
    prediction = model.predict(data_scaled)
    
    # Display result with color
    if prediction[0] == 1:
        st.error("⚠️ High risk of diabetes!")
    else:
        st.success("✅ Low risk of diabetes!")

# ------------------------------
# Optional visual / info
with st.expander("ℹ️ About this app"):
    st.write("""
    This app predicts the risk of diabetes based on user inputs. 
    Fill the values in the sidebar and click 'Predict' to see your risk.
    """)
