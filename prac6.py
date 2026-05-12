# Practical 6: Perform POS tagging in a given text file. Extract all nouns.
# Create and print a dictionary with frequency of parts of speech.
#
# HOW TO RUN:
# pip install nltk
# python practical6.py

import nltk
from collections import Counter

nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# Create the text file (so no dependency on external file in exam)
sample_text = """Machine learning is a branch of artificial intelligence.
Python is widely used for NLP and data science.
John works at Microsoft in India.
Natural language processing helps computers understand human language.
Deep learning models are trained on large datasets."""

with open("q6.txt", "w") as f:
    f.write(sample_text)

# Read text from file
with open("q6.txt", "r") as file:
    corpus = file.read()

print("Text from file:")
print(corpus)

# Tokenize
words = nltk.word_tokenize(corpus)

# POS Tagging
pos_tags = nltk.pos_tag(words)

print("\nPOS Tagged Words:")
print(pos_tags)

# Extract Nouns (NN, NNS, NNP, NNPS)
nouns = [word for word, tag in pos_tags if tag.startswith('NN')]
print("\nNouns in the text:")
print(nouns)

# POS Frequency Dictionary
pos_counts = Counter(tag for word, tag in pos_tags)
print("\nPOS Frequency Dictionary:")
print(dict(pos_counts))
