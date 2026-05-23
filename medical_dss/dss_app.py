import streamlit as st
import requests

st.title("Medical Decision System")

#Input fields
fever = st.selectbox("Do you fever?", [0, 1])
cough = st.selectbox("Do you have cough?", [0, 1])
shortness_of_breath = st.selectbox("Do you have shortness of breath?", [0, 1])
chest_pain = st.selectbox("Do you have chest pain?", [0, 1])

#Predict button
if st.button("Predict Illness"):
    data = {
        "fever": fever,
        "cough": cough,
        "shortness_of_breath": shortness_of_breath,
        "chest_pain": chest_pain
    }
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predict illness: {result['prediction']}")
        else:
            st.error("Server returned an error. Check Flask logs.")
    except Exception as e:
        st.error(f"Error connecting to Flask API: {e}")
