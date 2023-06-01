#!/usr/bin/env python3
"""0-initialize
Initializes cluster centroids for K means
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Parameters:
    X (numpy.ndarray): The dataset that will be used
    for K-means clustering. Shape (n, d).
    k (int): The number of clusters.

    Returns:
    numpy.ndarray: The initialized centroids
    for each cluster. Shape (k, d).
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None

    n, d = X.shape

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    centroids = np.random.uniform(min_vals, max_vals, (k, d))

    return centroids
