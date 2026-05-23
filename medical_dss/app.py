import streamlit as st
import joblib
import pandas as pd

#Load trained model
model = joblib.load('dss_model.pkl')

st.title("Medical DSS - Prediction App")
st.write("Enter patient symptoms to get a predicted illness")

#Input fields matching your model
fever = st.selectbox("Fever?", [0,1], format_func=lambda x: "Yes" if x == 1 else "No")
cough = st.selectbox("cough?", [0,1], format_func=lambda x: "Yes" if x == 1 else "No")
shortness_of_breath = st.selectbox("Shortness of breath?", [0,1], format_func=lambda x: "Yes" if x == 1 else"No")
chest_pain = st.selectbox("Chest pain?", [0,1], format_func=lambda x: "Yes" if x == 1 else "No")

if st.button("Predict"):
    input_data = pd.DataFrame([[fever, cough, shortness_of_breath, chest_pain]],
                              columns=['fever', 'cough', 'shortness_of_breath', 'chest_pain'])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted illness: **{prediction}**")    