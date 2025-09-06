import streamlit as st
import numpy as np
import pickle

# ---------------------------
# Load your trained model
# ---------------------------
# Make sure you have saved your trained model as 'model.pkl'
# Example: pickle.dump(model, open("model.pkl", "wb"))
model = pickle.load(open("regmodel.pkl", "rb"))

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🏠 California Housing Price Prediction")

st.write("Enter the house details below to get the predicted median house value:")

# Input fields for all features
MedInc = st.number_input("Median Income (10k $)", min_value=0.0, max_value=20.0, step=0.1)
HouseAge = st.number_input("House Age (years)", min_value=1, max_value=100, step=1)
AveRooms = st.number_input("Average Number of Rooms", min_value=0.0, max_value=50.0, step=0.1)
AveBedrms = st.number_input("Average Number of Bedrooms", min_value=0.0, max_value=10.0, step=0.1)
Population = st.number_input("Population", min_value=1, max_value=50000, step=10)
AveOccup = st.number_input("Average Occupancy", min_value=0.0, max_value=10.0, step=0.1)
Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, step=0.01)
Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-113.0, step=0.01)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Price"):
    # Arrange features in same order as training
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

    prediction = model.predict(features)

    st.success(f"🏡 Estimated Median House Value: ${prediction[0]*100000:,.2f}")
    st.caption("Note: Prices are scaled back to USD (approx).")

