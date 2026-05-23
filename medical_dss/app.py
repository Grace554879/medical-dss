import streamlit as st
import joblib
import pandas as pd

#Load trained model
model = joblib.load('dss_model.pkl')

st.title("Medical DSS - Prediction App")
st.write("Enter patient symptoms to get a predicted illness")

#Input fields
fever = st.selectbox("Do you have fever?", [0,1], format_func=lambda x: "Yes" if x == 1 else "No")
cough = st.selectbox("Do you have cough?", [0,1], format_func=lambda x: "Yes" if x == 1 else "No")
shortness_of_breath = st.selectbox("Do you have shortness of breath?", [0,1], format_func=lambda x: "Yes" if x == 1 else"No")
chest_pain = st.selectbox("Do you have chest pain?", [0,1], format_func=lambda x: "Yes" if x == 1 else "No")

if st.button("Predict Illness"):
    input_data = pd.DataFrame([[fever, cough, shortness_of_breath, chest_pain]],
                              columns=['fever', 'cough', 'shortness_of_breath', 'chest_pain'])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted illness: **{prediction}**")    