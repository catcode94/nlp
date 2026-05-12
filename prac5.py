# Practical 5: Build a simple statistical language model - estimate unigram and
# bigram probabilities with add-one smoothing and compute probability of sentences.
#
# HOW TO RUN:
# pip install nltk
# python practical5.py

import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict

nltk.download('punkt_tab', quiet=True)

# Training sentences
sentences = [
    "I love pizza",
    "The weather is nice today",
    "Python is a great language",
    "I love coding",
    "pizza is good",
    "you look nice today"
]

# Preprocess
data = [word_tokenize(s.lower()) for s in sentences]

# Unigram counts
unigram_counts = defaultdict(int)
for sentence in data:
    for word in sentence:
        unigram_counts[word] += 1

total_words = sum(unigram_counts.values())
vocab_size = len(unigram_counts)

print("Unigram Counts:")
for word, count in unigram_counts.items():
    print(f"  {word}: {count}")
print(f"\nVocabulary Size: {vocab_size}")
print(f"Total Words: {total_words}")

# Unigram probabilities with add-one smoothing
print("\nUnigram Probabilities (with Add-One Smoothing):")
for word in unigram_counts:
    prob = (unigram_counts[word] + 1) / (total_words + vocab_size)
    print(f"  P({word}) = {prob:.4f}")

# Bigram counts
bigram_counts = defaultdict(int)
for sentence in data:
    for i in range(len(sentence) - 1):
        bigram_counts[(sentence[i], sentence[i+1])] += 1

print("\nBigram Counts:")
for bigram, count in bigram_counts.items():
    print(f"  {bigram}: {count}")

# Bigram probability without smoothing
def bigram_prob(w1, w2):
    if unigram_counts[w1] == 0:
        return 0
    return bigram_counts[(w1, w2)] / unigram_counts[w1]

# Bigram probability with add-one (Laplace) smoothing
def bigram_prob_smooth(w1, w2):
    return (bigram_counts[(w1, w2)] + 1) / (unigram_counts[w1] + vocab_size)

print("\nBigram Probabilities (Without Smoothing):")
print(f"  P(love | i)    = {bigram_prob('i', 'love'):.4f}")
print(f"  P(pizza | love) = {bigram_prob('love', 'pizza'):.4f}")
print(f"  P(pizza | i)   = {bigram_prob('i', 'pizza'):.4f}")

print("\nBigram Probabilities (With Add-One Smoothing):")
print(f"  P(love | i)    = {bigram_prob_smooth('i', 'love'):.4f}")
print(f"  P(pizza | love) = {bigram_prob_smooth('love', 'pizza'):.4f}")
print(f"  P(pizza | i)   = {bigram_prob_smooth('i', 'pizza'):.4f}")

# Sentence probability
def sentence_prob(sentence):
    tokens = word_tokenize(sentence.lower())
    prob = 1
    for i in range(len(tokens) - 1):
        prob *= bigram_prob(tokens[i], tokens[i+1])
    return prob

def sentence_prob_smooth(sentence):
    tokens = word_tokenize(sentence.lower())
    prob = 1
    for i in range(len(tokens) - 1):
        prob *= bigram_prob_smooth(tokens[i], tokens[i+1])
    return prob

test_sentences = ["I love pizza", "love pizza you", "pizza is nice"]

print("\nSentence Probabilities (Without Smoothing):")
for s in test_sentences:
    print(f"  '{s}': {sentence_prob(s):.6f}")

print("\nSentence Probabilities (With Add-One Smoothing):")
for s in test_sentences:
    print(f"  '{s}': {sentence_prob_smooth(s):.6f}")
