import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

from preprocessing import preprocess_data, vectorize_text

# Load data
fake_df = pd.read_csv('Data/Fake.csv')
real_df = pd.read_csv('Data/True.csv')

# Add labels
fake_df['label'] = 1  # Fake
real_df['label'] = 0  # Real

# Combine and shuffle
df = pd.concat([fake_df, real_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocess
df = preprocess_data(df)

# Vectorize
X = vectorize_text(df['content'])
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
os.makedirs('Model', exist_ok=True)
with open('Model/logreg_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))
