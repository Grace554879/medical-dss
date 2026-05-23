import streamlit as st
import joblib
import pandas as pd

#page config
st.set_page_config(page_title="Medical DSS", page_icon="🏥", layout="centered")

#Load trained model
model = joblib.load('dss_model.pkl')

#Header 
st.title("🏥Medical DSS - Prediction App")
st.write("Enter patient symptoms to get a predicted illness")

st.divider()

#Use columns for better layout
col1, col2 = st.columns(2)

with col1:
    fever = st.selectbox("🌡️fever?", [0,1], format_func=lambda x: "Yes" if x == 1 else "No")
    cough = st.selectbox("😵Do you have cough?", [0,1], format_func=lambda x: "Yes" if x == 1 else "No")
with col2:
    shortness_of_breath = st.selectbox("🥲Do you have shortness of breath?", [0,1], format_func=lambda x: "Yes" if x == 1 else"No")
    chest_pain = st.selectbox("🫁Do you have chest pain?", [0,1], format_func=lambda x: "Yes" if x == 1 else "No")

if st.button("🩺Predict Illness"):
    input_data = pd.DataFrame([[fever, cough, shortness_of_breath, chest_pain]],
                              columns=['fever', 'cough', 'shortness_of_breath', 'chest_pain'])
    prediction = model.predict(input_data)[0]

    #Nicer result display
    st.success(f"Predicted illness: **{prediction}**") 
    st.caption("AI-assisted suggestion. Consult a medical professional for diagnosis")  
 