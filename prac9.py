# =============================================================================
# Practical 9: Build and train a text classifier using Keras (LSTM)
#              Dataset: IMDB 50K Movie Reviews
# =============================================================================
# HOW TO RUN:
# Step 1 (run once in terminal):
#         pip install pandas numpy scikit-learn tensorflow kagglehub
# Step 2: python practical9.py
# =============================================================================

import subprocess, sys

# Auto-install required libraries if missing
for package in ["pandas", "numpy", "scikit-learn", "tensorflow", "kagglehub"]:
    try:
        __import__(package if package != "scikit-learn" else "sklearn")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import re
import numpy as np
import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# -----------------------------------------------------------------------------
# 1. Auto Download Dataset
# -----------------------------------------------------------------------------
print("Downloading IMDB dataset... (cached after first run)")
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
print(f"Dataset ready at: {path}")

csv_file = None
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_file = os.path.join(path, file)
        break

if csv_file is None:
    raise FileNotFoundError("CSV file not found in downloaded dataset.")

# -----------------------------------------------------------------------------
# 2. Load Dataset
# -----------------------------------------------------------------------------
df = pd.read_csv(csv_file)
print(f"\nDataset loaded! Total reviews: {len(df)}")
print(df.head(3))

# Use only 10,000 rows to keep training fast (under 5 mins in exam)
df = df.sample(10000, random_state=42).reset_index(drop=True)
print(f"\nUsing {len(df)} reviews for faster training in exam")

# -----------------------------------------------------------------------------
# 3. Clean Text
# -----------------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)           # remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # remove punctuation/numbers
    return text

df['review'] = df['review'].apply(clean_text)

# -----------------------------------------------------------------------------
# 4. Encode Labels
# -----------------------------------------------------------------------------
encoder = LabelEncoder()
df['sentiment'] = encoder.fit_transform(df['sentiment'])  # positive=1, negative=0

print("\nSentiment distribution:")
print(df['sentiment'].value_counts())

# -----------------------------------------------------------------------------
# 5. Train / Test Split
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'],
    test_size=0.2, random_state=42
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# -----------------------------------------------------------------------------
# 6. Tokenization + Padding
# -----------------------------------------------------------------------------
vocab_size = 5000
max_len    = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq  = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad  = pad_sequences(X_test_seq,  maxlen=max_len, padding='post')

print(f"\nTraining data shape : {X_train_pad.shape}")
print(f"Testing data shape  : {X_test_pad.shape}")

# -----------------------------------------------------------------------------
# 7. Build Keras LSTM Model
# -----------------------------------------------------------------------------
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1,  activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# -----------------------------------------------------------------------------
# 8. Train Model
# -----------------------------------------------------------------------------
print("\nTraining model...")
history = model.fit(
    X_train_pad, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# -----------------------------------------------------------------------------
# 9. Evaluate Model
# -----------------------------------------------------------------------------
loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

# -----------------------------------------------------------------------------
# 10. Custom Review Predictions
# -----------------------------------------------------------------------------
def predict_review(text):
    cleaned = clean_text(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=max_len, padding='post')
    result  = model.predict(padded, verbose=0)[0][0]
    label   = "Positive" if result > 0.5 else "Negative"
    return label, result

print("\nCustom Review Predictions:")
print("-" * 50)

test_reviews = [
    "This movie was absolutely amazing and fantastic!",
    "This was the worst movie I have ever seen.",
    "The story was okay but acting was great.",
    "Completely boring and waste of time."
]

for review in test_reviews:
    label, score = predict_review(review)
    print(f"Review    : {review}")
    print(f"Sentiment : {label}  (confidence: {score:.2f})")
    print("-" * 50)
