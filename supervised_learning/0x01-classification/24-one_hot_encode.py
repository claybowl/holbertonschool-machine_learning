#!/usr/bin/env python3
"""module 24-one_hot_encode.py
converts a numeric label vector into a one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix"""
    if type(Y) is not np.ndarray:
        return None

    m = Y.shape[0]
    try:
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1

        return one_hot
    except Exception:
        return None
