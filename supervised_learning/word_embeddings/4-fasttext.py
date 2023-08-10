#!/usr/bin/env python3
"""module 4-fasttext.py
"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """function fasttext_model"""
    model = FastText(sentences=sentences,
                     vector_size=size,
                     min_count=min_count,
                     negative=negative,
                     window=window,
                     sg=0 if cbow else 1,
                     epochs=iterations,
                     seed=seed,
                     workers=workers)
    return model
