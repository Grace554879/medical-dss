import streamlit as st
import joblib
import pandas as pd
import shap

#Load trained model
model = joblib.load('dss_model.pkl')

#Now create explainer after model exists
explainer = shap.Explainer(model)

# page config
st.set_page_config(page_title="Medical DSS", page_icon="🏥", layout="centered")

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

    #Get prediction and confidence
    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data).max() *100

    st.success(f"Predicted illness: {prediction[0]}")
    st.info(f"Confidence: {confidence:.2f}%")    
 
    #SHAP explanation
    shap_values = explainer(input_data)
    st.subheader("Symptom contribution to prediction")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    #Plot using the values array directly
    shap.plots.bar(
        shap.Explanation(
            values=shap_values.values[0],
            feature_names=input_data.columns.tolist()
       ),
       show=False,
       ax=ax
    )
    st.pyplot(fig)