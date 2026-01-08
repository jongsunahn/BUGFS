import math
from collections import Counter
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    return [w for w in words if w.isalnum() and w not in stop_words]

def compute_idf(corpus):
    idf = {}
    N = len(corpus)
    
    for document in corpus:
        words = set(document)
        for word in words:
            if word in idf:
                idf[word] += 1
            else:
                idf[word] = 1
    
    for word, freq in idf.items():
        idf[word] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
    
    return idf

def compute_average_length(corpus):
    total_length = sum(len(document) for document in corpus)
    return total_length / len(corpus)

def get_bm25_vectors(corpus):
    processed_corpus = [preprocess_text(doc) for doc in corpus]
    idf = compute_idf(processed_corpus)
    avg_length = compute_average_length(processed_corpus)
    
    document_vectors = {}
    for doc_,doc in zip(corpus,processed_corpus):
        vector = []
        document_counts = Counter(doc)
        document_length = len(doc)
        
        for term, idf_value in idf.items():
            tf = document_counts.get(term, 0)
            numerator = idf_value * tf * (1.5 + 1)
            denominator = tf + 1.5 * (1 - 0.75 + 0.75 * (document_length / avg_length))
            vector.append(numerator / denominator)
        
        document_vectors[doc_] = vector
    
    return document_vectors

# # Example usage
# corpus = [
#     "This is the first document.",
#     "This document is the second document.",
#     "And this is the third one.",
#     "Is this the first document?",
# ]

# bm25_vectors = get_bm25_vectors(corpus)
# print("BM25 Vectors:")
# print(bm25_vectors)