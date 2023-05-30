#!/usr/bin/env python3
"""Module 1-pca
Function pca that performs PCA on a
dataset and returns the transformed
version of the dataset.
"""
import numpy as np


def pca(X, ndim):
    """Function that performs PCA on a dataset"""
    # Compute mean of data and center
    X_mean = X - np.mean(X, axis=0)
    # Compute the SVD of X
    U, S, V = np.linalg.svd(X_mean)
    # Return the weights matrix
    return V.T[:, :ndim]
