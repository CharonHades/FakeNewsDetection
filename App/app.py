import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

<<<<<<< HEAD
# Load model and vectorizer
model = joblib.load('Models/logreg_model.pkl')
vectorizer = joblib.load('Models/tfidf_vectorizer.pkl')
=======
# Load the pre-trained model and vectorizer
try:
    model = joblib.load('Model/logreg_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")

vectorizer = joblib.load('Model/tfidf_vectorizer.pkl')
>>>>>>> 42477b71eeb878d97da3c69fb0ef1ade848fa8e2

# Optional: Load test data for CM (make sure this is consistent with training/test split)
try:
    test_data = pd.read_csv("Data/test_data.csv")  # You can create this from your train/test split
    X_test_vectorized = vectorizer.transform(test_data["content"])
    y_test = test_data["label"]
    y_pred = model.predict(X_test_vectorized)
except Exception as e:
    y_test = y_pred = None
    print("Test data not loaded:", e)

# Function to predict
def predict(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0][1]
    return prediction, probability

# Function to show confusion matrix
def show_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

# UI
st.title("📰 Fake News Detection")
st.markdown("Check whether the news content is real or fake.")

# Input method
input_method = st.radio("Choose input method", ["Enter Text", "Upload Text File"])

text_input = ""
if input_method == "Enter Text":
    text_input = st.text_area("Enter news text here:")
elif input_method == "Upload Text File":
    uploaded_file = st.file_uploader("Choose a file", type="txt")
    if uploaded_file:
        text_input = uploaded_file.read().decode("utf-8")

# Predict
if st.button("Detect"):
    if text_input.strip():
        prediction, confidence = predict(text_input)
        label = "Real" if prediction == 1 else "Fake"

        st.markdown(f"### 🧾 Prediction: **{label}**")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

        # Optional: Show confusion matrix
        if y_test is not None and y_pred is not None:
            st.markdown("### 📊 Model Performance (Test Set)")
            show_confusion_matrix(y_test, y_pred)
    else:
        st.warning("Please enter or upload some text to analyze.")
