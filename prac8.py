# =============================================================================
# Practical 8: Classify movie reviews as positive or negative
#              using IMDB dataset of 50K movie reviews
# =============================================================================
# HOW TO RUN:
# Step 1 (run once in terminal):
#         pip install pandas scikit-learn
# Step 2: Download dataset CSV from:
#         https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
#         Place "IMDB Dataset.csv" in the SAME folder as this script
# Step 3: python practical8.py
# =============================================================================

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------------------------------------------
# Load Dataset
# -----------------------------------------------------------------------------
df = pd.read_csv("IMDB Dataset.csv")
print("Dataset loaded successfully!")
print(f"Total reviews: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nSample data:")
print(df.head(3))

# -----------------------------------------------------------------------------
# Text Cleaning
# -----------------------------------------------------------------------------
def clean_text(text):
    text = text.lower()                          # lowercase
    text = re.sub(r'<.*?>', '', text)            # remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)     # remove punctuation/numbers
    return text

df['review'] = df['review'].apply(clean_text)

# Encode labels: positive=1, negative=0
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("\nSentiment distribution:")
print(df['sentiment'].value_counts())

# -----------------------------------------------------------------------------
# Train / Test Split
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'],
    test_size=0.2, random_state=42
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# -----------------------------------------------------------------------------
# TF-IDF Vectorization
# -----------------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"\nTF-IDF matrix shape (train): {X_train_tfidf.shape}")

# -----------------------------------------------------------------------------
# Train Logistic Regression Model
# -----------------------------------------------------------------------------
print("\nTraining model... (may take a minute)")
model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# -----------------------------------------------------------------------------
# Evaluate Model
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test_tfidf)

print("\nModel Evaluation")
print("=" * 40)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# -----------------------------------------------------------------------------
# Sample Predictions from test set
# -----------------------------------------------------------------------------
print("Sample Predictions (first 10 from test set):")
print("-" * 50)

sample_reviews = X_test[:10]
sample_true    = y_test[:10]
sample_pred    = model.predict(vectorizer.transform(sample_reviews))

for review, true, pred in zip(sample_reviews, sample_true, sample_pred):
    print(f"Review    : {review[:50]}...")
    print(f"Actual    : {'Positive' if true == 1 else 'Negative'}")
    print(f"Predicted : {'Positive' if pred == 1 else 'Negative'}")
    print("-" * 50)

# -----------------------------------------------------------------------------
# Custom Review Prediction
# -----------------------------------------------------------------------------
print("\nCustom Review Predictions:")
test_reviews = [
    "This movie was absolutely amazing!",
    "This movie was exceptionally bad.",
    "The story was boring and too long.",
    "One of the best films I have ever seen."
]

for review in test_reviews:
    cleaned  = clean_text(review)
    tfidf    = vectorizer.transform([cleaned])
    result   = model.predict(tfidf)[0]
    label    = "Positive" if result == 1 else "Negative"
    print(f"  Review    : {review}")
    print(f"  Sentiment : {label}")
    print()
