import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the pre-trained model and vectorizer
try:
    model = joblib.load('Model/logreg_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")

vectorizer = joblib.load('Models/tfidf_vectorizer.pkl')

# Function to get the prediction
def predict(text):
    # Transform the input text using the same vectorizer used during training
    text_vectorized = vectorizer.transform([text])
    
    # Predict the label (Real or Fake)
    prediction = model.predict(text_vectorized)
    probability = model.predict_proba(text_vectorized)  # Get prediction probabilities
    
    return prediction[0], probability[0][1]  # Return the predicted label and confidence score

# Function to show confusion matrix
def show_confusion_matrix():
    # Sample confusion matrix (you can use your model's confusion matrix on test data)
    y_true = [1, 0, 1, 1, 0, 1, 0, 0]  # Example true labels (Real=1, Fake=0)
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0]  # Example predicted labels (Real=1, Fake=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot the confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title('Confusion Matrix')
    st.pyplot(fig)

# Streamlit app interface
st.title('Fake News Detection')
st.markdown("Upload or enter the text for prediction:")

# Text input field or file upload
input_method = st.radio("Choose input method", ["Enter Text", "Upload Text File"])

if input_method == "Enter Text":
    # Text input
    text_input = st.text_area("Enter news text here:")
elif input_method == "Upload Text File":
    # File upload
    uploaded_file = st.file_uploader("Choose a file", type="txt")
    if uploaded_file is not None:
        text_input = uploaded_file.read().decode('utf-8')
    else:
        text_input = ""

# When user clicks "Detect"
if st.button('Detect'):
    if text_input:
        # Get the prediction and confidence
        prediction, confidence = predict(text_input)
        
        # Display the prediction
        if prediction == 1:
            st.success(f"Prediction: **Real** News")
        else:
            st.error(f"Prediction: **Fake** News")
        
        # Show confidence score
        st.write(f"Confidence: {confidence * 100:.2f}%")
        
        # Show confusion matrix (this can be optional or added for more insights)
        st.write("Confusion Matrix Sample:")
        show_confusion_matrix()
    else:
        st.warning("Please enter text or upload a file to detect.")
