# app/app.py

import streamlit as st
import pandas as pd
import joblib
import sklearn 

# Load model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")


st.title("🧠 Agent Performance Optimization Engine")
st.write("Predict if a support agent is a High or Low performer.")

# Input fields
category = st.selectbox("Issue Category", encoders['category'].classes_)
sub_category = st.selectbox("Sub-Category", encoders['sub-category'].classes_)
agent_name = st.selectbox("Agent Name", encoders['agent_name'].classes_)
tenure_bucket = st.selectbox("Tenure", encoders['tenure_bucket'].classes_)
agent_shift = st.selectbox("Shift", encoders['agent_shift'].classes_)
csat_score = st.slider("CSAT Score", 1, 5, 3)

# Encode inputs
input_data = pd.DataFrame({
    'category': [encoders['category'].transform([category])[0]],
    'sub-category': [encoders['sub-category'].transform([sub_category])[0]],
    'agent_name': [encoders['agent_name'].transform([agent_name])[0]],
    'tenure_bucket': [encoders['tenure_bucket'].transform([tenure_bucket])[0]],
    'agent_shift': [encoders['agent_shift'].transform([agent_shift])[0]],
    'csat_score': [csat_score]
})

# Scale
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]

st.subheader("Prediction:")
if prediction == 1:
    st.success("✅ High Performer")
else:
    st.error("⚠️ Low Performer")

