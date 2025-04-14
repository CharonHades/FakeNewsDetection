import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def clean_text(text):
    """
    Clean input text by:
    - Removing punctuation
    - Lowercasing
    - Removing digits and URLs
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def preprocess_data(df):
    """
    Apply text cleaning to a DataFrame containing 'title' and 'text' columns.
    """
    df['content'] = df['title'] + ' ' + df['text']
    df['content'] = df['content'].apply(clean_text)
    return df[['content', 'label']]

def vectorize_text(corpus, vectorizer_path='Model/tfidf_vectorizer.pkl'):
    """
    Vectorize text using TF-IDF. Saves the vectorizer to disk.
    """
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = tfidf.fit_transform(corpus)

    # Save vectorizer
    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf, f)

    return X

def load_vectorizer(vectorizer_path='Model/tfidf_vectorizer.pkl'):
    """
    Load the saved TF-IDF vectorizer.
    """
    with open(vectorizer_path, 'rb') as f:
        tfidf = pickle.load(f)
    return tfidf
