import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("chd_model.pkl")

st.title("Heart Disease (CHD) Risk Predictor")
st.write("Enter patient details below to predict 10-year CHD risk.")

# --- Input fields ---
st.header("Patient Details")

male = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
age = st.number_input("Age", min_value=1, max_value=120, value=45)
education = st.selectbox("Education Level", options=[1.0, 2.0, 3.0, 4.0])
currentSmoker = st.selectbox("Current Smoker?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
cigsPerDay = st.number_input("Cigarettes per Day", min_value=0.0, max_value=70.0, value=0.0)
BPMeds = st.selectbox("On BP Medication?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
prevalentStroke = st.selectbox("History of Stroke?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
prevalentHyp = st.selectbox("Hypertension?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
diabetes = st.selectbox("Diabetes?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
totChol = st.number_input("Total Cholesterol", min_value=100.0, max_value=600.0, value=200.0)
sysBP = st.number_input("Systolic Blood Pressure", min_value=80.0, max_value=300.0, value=120.0)
diaBP = st.number_input("Diastolic Blood Pressure", min_value=40.0, max_value=200.0, value=80.0)
BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
heartRate = st.number_input("Heart Rate", min_value=30.0, max_value=200.0, value=75.0)
glucose = st.number_input("Glucose", min_value=40.0, max_value=400.0, value=90.0)

# --- Predict button ---
if st.button("Predict CHD Risk"):

    # Build a DataFrame exactly like the model saw during training
    input_data = pd.DataFrame([{
        "male": male,
        "age": age,
        "education": education,
        "currentSmoker": currentSmoker,
        "cigsPerDay": cigsPerDay,
        "BPMeds": BPMeds,
        "prevalentStroke": prevalentStroke,
        "prevalentHyp": prevalentHyp,
        "diabetes": diabetes,
        "totChol": totChol,
        "sysBP": sysBP,
        "diaBP": diaBP,
        "BMI": BMI,
        "heartRate": heartRate,
        "glucose": glucose
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # probability of class 1 (CHD risk)

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠ High Risk of CHD (probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk of CHD (probability: {probability:.2%})")