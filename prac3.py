# Practical 3: List the most common words (with their frequency)
# in a given text excluding stopwords.
#
# HOW TO RUN:
# pip install nltk
# python practical3.py

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Text corpus
corpus = """
The sky is blue and beautiful. Love this blue and beautiful sky!
The quick brown fox jumps over the lazy dog.
A king's breakfast has sausages, ham, bacon, eggs.
I love green eggs, ham, sausages and bacon!
The brown fox is quick and the blue dog is lazy!
The sky is very blue and the sky is very beautiful.
The dog is lazy but the brown fox is quick!
Toast and beans today.
"""

# Tokenize, lowercase, remove punctuation and stopwords
stop_words = set(stopwords.words('english'))
tokens = word_tokenize(corpus.lower())
filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

# Count frequencies
word_freq = Counter(filtered_tokens)

# Top 10 most common words
print("Top 10 Most Common Words (excluding stopwords):\n")
print(f"{'Word':<15} {'Frequency'}")
print("-" * 25)
for word, freq in word_freq.most_common(10):
    print(f"{word:<15} {freq}")
