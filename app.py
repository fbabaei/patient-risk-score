import streamlit as st
import pandas as pd
import pickle
import os

st.title("🏥 Patient Readmission Risk Predictor")

# ------------------------------
# Load model and scaler safely
# ------------------------------
MODEL_PATH = "model/model.pkl"
SCALER_PATH = "model/preprocessor.pkl"

def load_pickle(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        st.error(f"❌ Missing or empty file: {path}. Please retrain model first.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model = load_pickle(MODEL_PATH)
    scaler = load_pickle(SCALER_PATH)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Retrieve the feature names used during training
expected_features = list(getattr(scaler, "feature_names_in_", []))
if not expected_features:
    expected_features = [
        "age", "bmi", "blood_pressure", "cholesterol",
        "days_in_hospital", "num_prev_visits", "glucose_level"
    ]

# ------------------------------
# Collect user input
# ------------------------------
st.subheader("Enter Patient Information")

age = st.slider("Age", 18, 100, 45)
bmi = st.number_input("BMI", 10.0, 50.0, 24.5)
blood_pressure = st.number_input("Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol Level", 100, 400, 200)
days_in_hospital = st.number_input("Days in Hospital", 1, 30, 5)
num_prev_visits = st.number_input("Previous Hospital Visits", 0, 20, 2)
glucose_level = st.number_input("Glucose Level", 60, 300, 110)

input_data = {
    "age": age,
    "bmi": bmi,
    "blood_pressure": blood_pressure,
    "cholesterol": cholesterol,
    "days_in_hospital": days_in_hospital,
    "num_prev_visits": num_prev_visits,
    "glucose_level": glucose_level,
}

X = pd.DataFrame([input_data])

# ------------------------------
# Validation: check columns
# ------------------------------
missing = [f for f in expected_features if f not in X.columns]
if missing:
    st.error(f"❌ Missing features: {missing}. Please update app input fields.")
    st.stop()

# ------------------------------
# Make prediction
# ------------------------------
if st.button("Predict Readmission Risk"):
    try:
        X_scaled = scaler.transform(X[expected_features])
        prob = model.predict_proba(X_scaled)[0][1]
        st.success(f"Predicted Readmission Probability: {prob:.2%}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
