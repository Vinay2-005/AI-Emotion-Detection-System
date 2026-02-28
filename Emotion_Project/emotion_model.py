# STEP 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# STEP 2: Load dataset
data = pd.read_csv("emotion_data.csv")

# STEP 3: Separate input and output
X = data["Comment"]
y = data["Emotion"]

# STEP 4: Convert text to numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# STEP 5: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# STEP 6: Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# STEP 7: Test the model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))


# STEP 8: Initialize stress counter and history
stress_count = 0
history = []

print("\nEmotion Detection System Started (Type 'exit' to stop)\n")


# STEP 9: Take user input and predict emotion
while True:
    text = input("Enter a sentence: ")

    if text.lower() == "exit":
        print("\nSession Ended.")
        print("Emotion History:", history)
        break

    # Predict emotion
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    emotion = prediction[0]

    print("Predicted Emotion:", emotion)

    # Save history
    history.append(emotion)

    # TASK RECOMMENDATION SYSTEM
    if emotion == "joy":
        print("Recommended Task: Assign creative or challenging work")

    elif emotion == "sadness":
        print("Recommended Task: Give light workload and supportive tasks")

    elif emotion == "anger":
        print("Recommended Task: Suggest short break or stress relief activity")

    elif emotion == "fear":
        print("Recommended Task: Provide team-based supportive tasks")

    else:
        print("Recommended Task: Assign normal work tasks")

    # STRESS DETECTION SYSTEM
    if emotion in ["sadness", "anger", "fear"]:
        stress_count += 1
    else:
        stress_count = 0

    # ALERT IF STRESS CONTINUES
    if stress_count >= 3:
        print("⚠ ALERT: Employee may be under continuous stress! Notify HR immediately.")