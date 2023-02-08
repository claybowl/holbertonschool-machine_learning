#!/usr/bin/env python3
"""module 25-one_hot_decode.py
converts a numeric label vector into a one-hot matrix
"""
import numpy as np


def one_hot_decode(one_hot):
    try:
        m = one_hot.shape[1]
        return np.argmax(one_hot, axis=0).reshape(m,)
    except:
        return None
