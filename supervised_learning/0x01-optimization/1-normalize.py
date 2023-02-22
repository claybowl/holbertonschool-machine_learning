"""Module 1-normalize
subtracts the mean of each feature
from the feature values, and
divides the result by the standard deviation
of the feature
"""
import numpy as np


def normalize(X, m, s):
    """Normalizes the feature values"""
    X_norm = (X - m) / s
    return X_norm
