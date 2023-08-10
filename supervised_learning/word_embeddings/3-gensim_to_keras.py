#!/usr/bin/env python3
"""module 3-gensim_to_keras
"""


def gensim_to_keras(model):
    """function gensim_to_keras"""
    return model.wv.get_keras_embedding(True)
