import streamlit as st
import pandas as pd
import pickle

# Load model and preprocessor
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/preprocessor.pkl", "rb"))

st.title("🏥 Patient Readmission Risk Predictor")

# User input fields
age = st.slider("Age", 18, 100, 45)
bmi = st.number_input("BMI", 10.0, 50.0, 24.5)
blood_pressure = st.number_input("Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol Level", 100, 400, 200)
days_in_hospital = st.number_input("Days in Hospital", 1, 30, 5)

# Make prediction
if st.button("Predict Readmission Risk"):
    X = pd.DataFrame([[age, bmi, blood_pressure, cholesterol, days_in_hospital]],
                     columns=["age", "bmi", "blood_pressure", "cholesterol", "days_in_hospital"])
    X_scaled = scaler.transform(X)
    prediction = model.predict_proba(X_scaled)[0][1]
    st.success(f"Predicted Readmission Probability: {prediction:.2%}")
