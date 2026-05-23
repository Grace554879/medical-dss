import streamlit as st
import joblib
import pandas as pd

#Load trained model
model = joblib.load('dss_model.pkl')

st.title("Medical DSS - Prediction App")
st.write("Enter patient data to get a prediction")

#Example inputs - change these to match your model features
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
#Add more inputs based on what your model needs

if st.button("Predict"):
    input_df = pd.DataFrame([[feature1, feature2]]) #match your model columns
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {prediction}")    