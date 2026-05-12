# =============================================================================
# Practical 7: Identify and print named entities using NER for news headlines
# =============================================================================
# HOW TO RUN:
# Step 1 (run once in terminal):
#         pip install spacy
#         python -m spacy download en_core_web_sm
# Step 2: python practical7.py
# =============================================================================

import spacy
from collections import Counter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Collection of news headlines
news_headlines = [
    "Narendra Modi met Joe Biden in New York",
    "Microsoft launched a new AI tool in India",
    "Elon Musk announced Tesla expansion in Germany",
    "Google opened a new office in Bengaluru",
    "Apple unveiled the new iPhone in California",
    "United Nations discussed climate change in Paris",
    "Shah Rukh Khan attended an event in Mumbai",
    "NASA and ISRO signed a space exploration agreement",
    "Amazon acquired a startup in Seattle for 500 million dollars",
    "The Reserve Bank of India raised interest rates in March 2024",
]

# -----------------------------------------------
# Perform NER and print results
# -----------------------------------------------
print("Named Entity Recognition (NER) Results")
print("=" * 50)

all_labels = []

for headline in news_headlines:
    doc = nlp(headline)
    print(f"\nHeadline : {headline}")
    print("Entities :")
    for ent in doc.ents:
        print(f"  {ent.text}  -->  {ent.label_}  ({spacy.explain(ent.label_)})")
        all_labels.append(ent.label_)
    print("-" * 50)

# -----------------------------------------------
# Entity label frequency summary
# -----------------------------------------------
print("\nEntity Type Frequency:")
for label, count in Counter(all_labels).most_common():
    print(f"  {label} ({spacy.explain(label)}) : {count}")
