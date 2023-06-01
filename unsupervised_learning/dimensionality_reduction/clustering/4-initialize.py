#!/usr/bin/env python3
"""4-initialize
initializes variables for a Gaussian Mixture Mode
"""
import numpy as np


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.

    Parameters:
    X (numpy.ndarray): The data set. Shape (n, d).
    k (int): The number of clusters.

    Returns:
    tuple: (pi, m, S)
        pi is a numpy.ndarray of shape (k,) containing the priors for each cluster, initialized evenly.
        m is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster, initialized with K-means.
        S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices for each cluster, initialized as identity matrices.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None, None

    n, d = X.shape

    pi = np.full((k,), 1 / k)

    m, _ = kmeans(X, k)

    S = np.full((k, d, d), np.identity(d))

    return pi, m, S
