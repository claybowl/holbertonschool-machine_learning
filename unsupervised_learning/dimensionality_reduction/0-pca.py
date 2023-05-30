#!/usr/bin/env python3
"""Module 0-pca
Function that performs PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """Function that performs PCA on a dataset"""
    # Compute the SVD of X
    _, S, Vt = np.linalg.svd(X, full_matrices=False
    # Compute the variance explained by each component
    evr = S**2 / np.sum(S**2)
    # Calculate the cumsum
    cumsum_evr = np.cumsum(evr)
    # Calculate how many dim should maintain variance
    dimensions = np.argmax(cumsum_evr >= var) + 1
    # Return the weights matrix
    Weights_matrix = Vt[:dimensions + 1].T

    return Weights_matrix
