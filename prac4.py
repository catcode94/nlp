# Practical 4: Create the TF-IDF (Term Frequency - Inverse Document Frequency)
# Matrix for the given set of text documents.
#
# HOW TO RUN:
# pip install scikit-learn pandas
# python practical4.py

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample documents
documents = [
    "I Like Ice Cream",
    "He Likes Ice Cream",
    "She Likes Ice Cream",
    "They Like Ice Cream",
    "Them Like Kulfi"
]

# Create TF-IDF Vectorizer and fit
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature (term) names
terms = vectorizer.get_feature_names_out()

# Convert to array
matrix = tfidf_matrix.toarray()

# Display as readable DataFrame
df = pd.DataFrame(matrix, columns=terms, index=[f"Doc{i+1}" for i in range(len(documents))])

print("TF-IDF Matrix:\n")
print(df.round(4))
