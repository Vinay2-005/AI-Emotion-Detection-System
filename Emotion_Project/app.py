import streamlit as st
import pickle

# Load saved files
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("AI-Powered Emotion Detection System")

user_input = st.text_input("Enter your text here:")

if user_input:
    text_vec = vectorizer.transform([user_input])
    prediction = model.predict(text_vec)[0]

    st.subheader(f"Predicted Emotion: {prediction}")