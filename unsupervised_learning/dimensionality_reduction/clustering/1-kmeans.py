#!/usr/bin/env python3
"""1-kmeans
Performs Kmeans on a dataset
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Parameters:
    X (numpy.ndarray): The dataset. Shape (n, d).
    k (int): The number of clusters.
    iterations (int): The maximum number of iterations that should be performed.

    Returns:
    tuple: (C, clss)
        C is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster.
        clss is a numpy.ndarray of shape (n,) containing the index of the cluster in C that each data point belongs to.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    C = initialize(X, k)
    clss = None

    for _ in range(iterations):
        C_prev = np.copy(C)

        dist = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
        clss = np.argmin(dist, axis=0)

        for j in range(k):
            if (clss == j).any():
                C[j] = np.mean(X[clss == j], axis=0)
            else:
                C[j] = initialize(X, 1)

        if np.all(C_prev == C):
            return C, clss

    return C, clss
