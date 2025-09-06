import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load dataset (for demo only, don't include in real fraud system)
data = pd.read_csv("creditcard.csv")

# Load trained model
model = pickle.load(open("Frauddetectmodel.pkl", "rb"))

st.title("💳 Credit Card Fraud Detection")

# Let user pick a transaction
row_id = st.number_input("Enter Transaction ID (row number)", min_value=0, max_value=len(data)-1, step=1)

if st.button("🔍 Detect Fraud"):
    # Get row features (drop the target column 'Class')
    features = data.drop("Class", axis=1).iloc[row_id].values.reshape(1, -1)

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")


