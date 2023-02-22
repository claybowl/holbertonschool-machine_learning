"""Module 2-shuffle_data
Shuffles the data points of two
matrices the same way.
"""
import numpy as np


def shuffle_data(X, Y):
    """shuffles data points of two matrices"""
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
