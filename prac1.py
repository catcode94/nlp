# Practical 1: NLP Text Preprocessing
# Tokenization, Lowercase, Punctuation Removal, Stopword Filtration, Stemming, Lemmatization
#
# HOW TO RUN:
# pip install nltk spacy
# python -m spacy download en_core_web_sm
# python practical1.py

import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

corpus = """
Natural Language Processing (NLP) is one of the most important fields of Artificial Intelligence.
It helps computers interpret, understand, and generate human language.
NLP techniques are widely used in machine translation, sentiment analysis, and chatbots.
"""

# Tokenization
tokens = word_tokenize(corpus)
print("Tokens:\n", tokens)

# Lowercase + Punctuation Removal
tokens_lower = [word.lower() for word in tokens if word.isalpha()]
print("\nLowercase + Punctuation Removed:\n", tokens_lower)

# Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens_lower if word not in stop_words]
print("\nAfter Stopword Removal:\n", filtered_tokens)

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("\nStemmed Tokens:\n", stemmed_tokens)

# Lemmatization
nlp = spacy.load('en_core_web_sm')
doc = nlp(' '.join(filtered_tokens))
lemmatized_tokens = [token.lemma_ for token in doc]
print("\nLemmatized Tokens:\n", lemmatized_tokens)

# Summary
print("\n--- Summary ---")
print("Original tokens         :", len(tokens))
print("After lowercase + punct :", len(tokens_lower))
print("After stopword removal  :", len(filtered_tokens))
print("After stemming          :", len(stemmed_tokens))
print("After lemmatization     :", len(lemmatized_tokens))
