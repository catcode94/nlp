import nltk
import re
import pandas as pd
from nltk.corpus import gutenberg
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('gutenberg')
nltk.download('punkt')

# Load corpus
text = gutenberg.raw('austen-emma.txt')
sentences = nltk.sent_tokenize(text)

# Take first 10 sentences
corpus = sentences[:10]

# Preprocessing function
def custom_preprocessor(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc)
    return doc.lower()

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    preprocessor=custom_preprocessor,
    stop_words='english'
)

# Create TF-IDF Matrix
result = tfidf.fit_transform(corpus)

# Convert to DataFrame
df = pd.DataFrame(
    result.toarray(),
    columns=tfidf.get_feature_names_out()
)

# Display Matrix
print("TF-IDF Matrix:\n")
print(df)
