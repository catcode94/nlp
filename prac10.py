# =============================================================================
# Practical 10: Character-Based Text Generation using LSTM
# Question: Generate text using a character-based model using an appropriate
#           dataset. Given a sequence of characters (e.g., "Shakespear"),
#           train a model to predict the next character (e.g., "e").
#
# HOW TO RUN:
#   Step 1: pip install numpy tensorflow
#   Step 2: python practical10.py
#   Note  : Runs in ~2-4 minutes. No internet needed (uses built-in dataset).
#           To use Gutenberg dataset instead, set USE_GUTENBERG = True (needs internet).
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ── Toggle: Set True only if teacher asks for Gutenberg ──────────────────────
USE_GUTENBERG = False
# ─────────────────────────────────────────────────────────────────────────────

# ----------------------------------------------------------------------------
# STEP 1: Load Dataset
# ----------------------------------------------------------------------------

if USE_GUTENBERG:
    import requests, re
    print("Loading dataset from Project Gutenberg (Romeo and Juliet)...")
    url = "https://www.gutenberg.org/cache/epub/1513/pg1513.txt"
    raw = requests.get(url, timeout=15).text
    start = raw.find("THE TRAGEDY OF ROMEO AND JULIET")
    end   = raw.find("End of the Project Gutenberg")
    text  = raw[start:end] if start != -1 else raw
    text  = re.sub(r'[^a-z\s]', '', text[:5000].lower())  # keep small = fast
    print(f"Gutenberg text loaded. Using first 5000 characters.\n")
else:
    text = """
shakespeare was a famous writer.
machine learning is powerful.
natural language processing is interesting.
deep learning models predict sequences.
python is widely used in artificial intelligence.
students study neural networks in nlp.
character prediction uses lstm models.
tensorflow helps build deep learning systems.
data science and ai are growing fields.
language models learn character patterns.
"""
    text = text.lower()

# ----------------------------------------------------------------------------
# STEP 2: Character Mappings
# ----------------------------------------------------------------------------

chars       = sorted(list(set(text)))
n_chars     = len(chars)
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

print(f"Total characters in text : {len(text)}")
print(f"Unique characters        : {n_chars}")
print(f"Vocabulary               : {''.join(chars)}\n")

# ----------------------------------------------------------------------------
# STEP 3: Prepare Sequences  (seq_length = 10, predict 11th character)
# ----------------------------------------------------------------------------

seq_length = 10
X, y = [], []

for i in range(len(text) - seq_length):
    seq_in  = text[i : i + seq_length]
    seq_out = text[i + seq_length]
    X.append([char_to_int[c] for c in seq_in])
    y.append(char_to_int[seq_out])

X = np.array(X)
y = np.array(y)

print(f"Total training samples   : {len(X)}")

# Normalize + reshape for LSTM input → (samples, timesteps, 1)
X = X / float(n_chars)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print(f"X shape                  : {X.shape}")
print(f"y shape                  : {y.shape}\n")

# ----------------------------------------------------------------------------
# STEP 4: Build LSTM Model
# ----------------------------------------------------------------------------

model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], 1)))
model.add(Dense(n_chars, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()
print()

# ----------------------------------------------------------------------------
# STEP 5: Train the Model
# ----------------------------------------------------------------------------

print("Training model...\n")

history = model.fit(X, y, epochs=250, batch_size=16, verbose=1)

print(f"\nTraining Complete!")
print(f"Final Loss     : {history.history['loss'][-1]:.4f}")
print(f"Final Accuracy : {history.history['accuracy'][-1]*100:.2f}%\n")

# ----------------------------------------------------------------------------
# STEP 6: Predict Next Character  (the actual exam output)
# ----------------------------------------------------------------------------

print("=" * 45)
print("        CHARACTER PREDICTION RESULTS")
print("=" * 45)

seed = "shakespear"   # → expected next char: 'e'

pattern    = [char_to_int[c] for c in seed]
x_input    = np.array(pattern) / float(n_chars)
x_input    = np.reshape(x_input, (1, seq_length, 1))

prediction = model.predict(x_input, verbose=0)
index      = np.argmax(prediction)
result     = int_to_char[index]

print(f"\n  Input Sequence         : '{seed}'")
print(f"  Predicted Next Char    : '{result}'")
print(f"  Confidence             : {prediction[0][index]*100:.2f}%")
print("=" * 45)
