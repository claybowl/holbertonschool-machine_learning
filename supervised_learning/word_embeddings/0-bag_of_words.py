#!/usr/bin/env python3
"""module 0-bag_of_words.py
"""
import string
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Function that creates a bag of words embedding matrix"""

    translator = str.maketrans('', '', string.punctuation)
    features = []

    # lowercase it all
    sentences = [i.lower() for i in sentences]
    sentences = [i.replace("'s", '') for i in sentences]

    # remove punctuation
    for i in range(len(sentences)):
        sentences[i] = sentences[i].translate(translator)

    corpus = sentences.copy()

    # split words up
    for elem in sentences:
        append = elem.split()
        features.extend(append)

    # filters the features by vocab passed in
    if vocab is not None:
        features = vocab

    # sorts by alphabetical order
    if vocab is None:
        features = sorted(list(set(features)))

    cv = CountVectorizer(vocabulary=features)
    embedding = cv.fit_transform(corpus).toarray()

    return embedding, features
