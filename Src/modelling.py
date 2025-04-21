import pandas as pd
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import preprocess_data, vectorize_text

# Paths
DATA_DIR = 'Data'
MODEL_DIR = 'Models'
MODEL_PATH = os.path.join(MODEL_DIR, 'logreg_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.csv')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Load data
fake_df = pd.read_csv(os.path.join(DATA_DIR, 'Fake.csv'))
real_df = pd.read_csv(os.path.join(DATA_DIR, 'True.csv'))

# Add labels: 0 = Fake, 1 = Real
fake_df['label'] = 0
real_df['label'] = 1

# Combine and shuffle
df = pd.concat([fake_df, real_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocess
df = preprocess_data(df)

# Vectorize (also saves the vectorizer)
X = vectorize_text(df['content'], vectorizer_path=VECTORIZER_PATH)
y = df['label']

# Train/test split (stratified to maintain label balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# Save model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

# Save test set for Streamlit app usage
test_df = df.iloc[y_test.index]
test_df.to_csv(TEST_DATA_PATH, index=False)

# Evaluate
y_pred = model.predict(X_test)

print("✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))
