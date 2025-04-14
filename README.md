# 📰 Fake News Detection Web App

This is a simple Fake News Detection system built with **Python**, **Scikit-learn**, and **Streamlit**. It uses a logistic regression model trained on real and fake news articles to classify user-provided text as either **Real** or **Fake**.

---

## 🚀 Features

- Upload a `.txt` file or enter news article text manually
- Predict whether the news is **Real** or **Fake**
- Display model confidence score
- Show a confusion matrix sample from validation

---

## 📁 Project Structure

FakeNewsDetection/ │ ├── Data/ # Raw data (Fake.csv and True.csv) ├── App/ # Streamlit app │ └── streamlit_app.py ├── Models/ # Saved model and vectorizer │ ├── logreg_model.pkl │ └── tfidf_vectorizer.pkl ├── Src/ # Preprocessing and training scripts │ ├── preprocessing.py │ └── train_model.py ├── README.md # You’re here! ├── requirements.txt # Project dependencies └── .gitignore # Ignored files for Git

---

## 🧠 Model Details

- **Algorithm:** Logistic Regression
- **Vectorizer:** TF-IDF
- **Training Data:** Combined dataset of real and fake news
- **Evaluation:** Accuracy score and Confusion Matrix

---

## 🛠️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/FakeNewsDetection.git
cd FakeNewsDetection

2. Create a Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Streamlit App
streamlit run App/streamlit_app.py

🧾 Dataset
This project uses the Fake and Real News Dataset from Kaggle:

Dataset https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets?resource=download

📊 Sample Output
Prediction: ✅ Real / ❌ Fake

Confidence: e.g., 87.34%

Confusion Matrix: Static visual from model validation

🙌 Acknowledgements
>Scikit-learn
>Streamlit
>Pandas
>Seaborn

📌 Future Improvements
Enable PDF or news URL input

Add ROC curve and AUC score

Use transformer-based models (e.g., BERT)

Live retraining with user feedback

👤 Author
Your Harsh Sharma
GitHub: github.com/CharonHades
Email: pd.harshsharma@gmail.com