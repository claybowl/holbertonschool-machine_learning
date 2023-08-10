#!/usr/bin/env python3
"""module 2-word2vec.py
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Function that creates and trains a gensim word2vec model"""
    model = Word2Vec(sentences=sentences,
                     vector_size=size,
                     window=window,
                     min_count=min_count,
                     negative=negative,
                     sg=0 if cbow else 1,
                     epochs=iterations,
                     seed=seed,
                     workers=workers)
    return model
