import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("xgb_calorie_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

st.title("Calories Burned Prediction App")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=25)
height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
duration = st.number_input("Exercise Duration (min)", min_value=1, max_value=300, value=30)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=220, value=100)
body_temp = st.number_input("Body Temp (°C)", min_value=30.0, max_value=45.0, value=37.0)  # <-- match column name

gender_val = 0 if gender.lower() == "male" else 1

# Ensure order matches: ['User_ID', 'Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
input_data = np.array([[ gender_val, age, height, weight, duration, heart_rate, body_temp]])

if st.button("Predict Calories Burned"):
    prediction = loaded_model.predict(input_data)
    st.success(f"Estimated Calories Burned: {prediction[0]:.2f}")




