#!/usr/bin/env python3
"""Module 0-pca
Function that performs PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """Function that performs PCA on a dataset"""
    # Compute the SVD of X
    _, _, V = np.linalg.svd(X)
    # Compute the variance explained by each component
    cumulative = np.cumsum(V**2) / np.sum(V**2)
    # number of princ comp that explain 'var' of the variance
    r = (np.argwhere(cumulative >= var))[0, 0]
    # Return the weights matrix
    return V.T[:, :r + 1]
