#!/usr/bin/env python3
"""module 24-one_hot_encode.py
converts a numeric label vector into a one-hot matrix
"""
import numpy as np


def one_hot_encode(Y, classes):
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        return None

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1

    return one_hot
