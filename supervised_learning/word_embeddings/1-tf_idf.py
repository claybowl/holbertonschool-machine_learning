#!/usr/bin/env python3
"""1-tf_idf.py
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """function that creates a TF-IDF embedding"""
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    vectorized = vectorizer.fit_transform(sentences)
    embeddings = vectorized.toarray()
    features = vectorizer.get_feature_names()

    return embeddings, features
