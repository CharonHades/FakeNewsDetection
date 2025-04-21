import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Try to load the model and vectorizer
try:
    model = joblib.load('Model/logreg_model.pkl')
    vectorizer = joblib.load('Model/tfidf_vectorizer.pkl')
except Exception as e:
    st.error(f"❌ Error loading model or vectorizer: {e}")
    st.stop()

# Try loading test data for evaluation (optional)
try:
    test_data = pd.read_csv("Data/test_data.csv")  # Make sure this file exists!
    X_test_vectorized = vectorizer.transform(test_data["content"])
    y_test = test_data["label"]
    y_pred = model.predict(X_test_vectorized)
except Exception as e:
    y_test = y_pred = None
    print("⚠️ Test data not loaded:", e)

# Prediction function
def predict(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0][1]
    return prediction, probability

# Confusion matrix display
def show_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)

# Streamlit UI
st.title("📰 Fake News Detection")
st.markdown("Determine whether a news article is **real** or **fake** using a trained ML model.")

# Choose input method
input_method = st.radio("Choose input method", ["Enter Text", "Upload Text File"])
text_input = ""

if input_method == "Enter Text":
    text_input = st.text_area("📝 Enter news content:")
elif input_method == "Upload Text File":
    uploaded_file = st.file_uploader("📄 Upload a .txt file", type="txt")
    if uploaded_file:
        text_input = uploaded_file.read().decode("utf-8")

# Prediction button
if st.button("🔍 Detect"):
    if text_input.strip():
        prediction, confidence = predict(text_input)
        label = "Real" if prediction == 1 else "Fake"

        st.markdown(f"### ✅ Prediction: **{label}**")
        st.markdown(f"**Confidence Score:** {confidence * 100:.2f}%")

        # Optional model performance
        if y_test is not None and y_pred is not None:
            st.markdown("---")
            st.markdown("### 📊 Model Performance (on test data)")
            show_confusion_matrix(y_test, y_pred)
    else:
        st.warning("⚠️ Please enter or upload text to get a prediction.")
