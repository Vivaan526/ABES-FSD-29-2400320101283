# train_model.py
"""
Train a simple spam classifier (TF-IDF + MultinomialNB) and save the model and vectorizer.
Expecting a CSV 'data/spam.csv' with at least two columns: 'label' and 'message'.
'label' should be 'spam' or 'ham' (or similar).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

DATA_PATH = "data/spam.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1) Load dataset
df = pd.read_csv(DATA_PATH, encoding='latin-1')  # encoding often needed for SMS dataset
# Keep only necessary columns - try to detect common column names
if 'v1' in df.columns and 'v2' in df.columns:
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df = df[['label', 'message']].dropna()

# 2) Simple label conversion: spam -> 1, ham -> 0
df['label_num'] = df['label'].map(lambda x: 1 if x.strip().lower() == 'spam' else 0)

# 3) Split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'],
                                                    test_size=0.2, random_state=42,
                                                    stratify=df['label_num'])

# 4) Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(strip_accents='unicode', lowercase=True,
                             stop_words='english', max_df=0.95, min_df=2)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5) Train classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6) Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7) Save model and vectorizer
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
joblib.dump(model, os.path.join(MODEL_DIR, "spam_model.pkl"))

print("Saved model and vectorizer to", MODEL_DIR)
