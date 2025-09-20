import streamlit as st
import pickle
import pandas as pd

# Load model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💳 Online Fraud Detection")

# Transaction type mapping (must match training)
type_mapping = {'CASH_OUT': 1,'PAYMENT': 2, 'CASH_IN': 3,'TRANSFER': 4, 'DEBIT': 5}

# Inputs
transaction_type = st.selectbox("Transaction Type", list(type_mapping.keys()))
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
oldbalance = st.number_input("Old Balance", min_value=0.0, value=1000.0)
newbalance = st.number_input("New Balance", min_value=0.0, value=900.0)

if st.button("Predict Fraud"):
    # Encode transaction type as numeric
    transaction_type_encoded = type_mapping[transaction_type]

    # Create input DataFrame in same order as training
    input_df = pd.DataFrame([[transaction_type_encoded, amount, oldbalance, newbalance]],
                            columns=['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig'])

    # Predict
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate")

