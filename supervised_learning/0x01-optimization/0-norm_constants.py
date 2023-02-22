"""Module 0-norm_constants
calculates the normalization (standardization) constants of a matrix
"""
import numpy as np


def normalization_constants(X):
    """calculates the normalization"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
