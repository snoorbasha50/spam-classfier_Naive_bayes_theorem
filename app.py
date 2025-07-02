import numpy as np
import streamlit as st
import joblib


##Load the model and vectorizer
model=joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📩 Spam vs Ham Classifier")
st.subheader("Built with Naive Bayes + NLP")



##Input take from user
user_input = st.text_area("Enter a message to classify:", height=150)


##Predict
if st.button("Predict"):
    if user_input.strip()=="":
       st.warning("Please enter a message.")
    else:
        vectorizer_input=vectorizer.transform([user_input])
        prediction=model.predict(vectorizer_input)[0]

        label = "🚫 Spam" if prediction == 1 else "✅ Not Spam (Ham)"
        st.success(f"Prediction: {label}")