import streamlit as st
import pickle
import pandas as pd
import os
from datetime import datetime

# -------------------------------
# Load Saved Model and Vectorizer
# -------------------------------
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("AI-Powered Employee Emotion & Task Optimization System")

employee_id = st.text_input("Enter Employee ID")

user_input = st.text_area("Enter Employee Feedback")

# -------------------------------
# Task Recommendation Function
# -------------------------------
def recommend_task(emotion):

    if emotion == "joy":
        return "Assign creative or high-priority work"

    elif emotion == "sadness":
        return "Give light workload and supportive tasks"

    elif emotion == "anger":
        return "Suggest short break or stress relief activity"

    elif emotion == "fear":
        return "Provide team-based supportive tasks"

    else:
        return "Assign normal work tasks"


# -------------------------------
# Predict Emotion
# -------------------------------
if st.button("Analyze Emotion"):

    if employee_id and user_input:

        # Convert text
        text_vec = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(text_vec)[0]

        st.subheader(f"Predicted Emotion: {prediction}")

        # -------------------------------
        # Task Recommendation
        # -------------------------------
        task = recommend_task(prediction)

        st.success(f"Recommended Task: {task}")

        # -------------------------------
        # Save History to CSV
        # -------------------------------
        file_name = "employee_mood_history.csv"

        new_data = pd.DataFrame({
            "Employee_ID": [employee_id],
            "Date": [datetime.now()],
            "Feedback": [user_input],
            "Emotion": [prediction]
        })

        if os.path.exists(file_name):
            new_data.to_csv(
                file_name,
                mode="a",
                header=False,
                index=False
            )
        else:
            new_data.to_csv(
                file_name,
                index=False
            )

        st.info("Emotion history saved successfully")

        # -------------------------------
        # Stress Detection
        # -------------------------------
        history = pd.read_csv(file_name)

        negative_emotions = ["sadness", "anger", "fear"]

        stress_count = len(
            history[
                (history["Employee_ID"] == employee_id)
                & (
                    history["Emotion"]
                    .isin(negative_emotions)
                )
            ]
        )

        # -------------------------------
        # HR Alert
        # -------------------------------
        if stress_count >= 3:
            st.error(
                "⚠ ALERT: Employee may be under continuous stress! Notify HR immediately."
            )

        # -------------------------------
        # Show Employee History
        # -------------------------------
        st.subheader("Employee Mood History")

        emp_history = history[
            history["Employee_ID"] == employee_id
        ]

        st.dataframe(emp_history)

    else:
        st.warning("Please enter Employee ID and feedback")
