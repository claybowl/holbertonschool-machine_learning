#!/usr/bin/env python3
"""1-kmeans
Performs Kmeans on a dataset
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


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Parameters:
    X (numpy.ndarray): The dataset. Shape (n, d).
    k (int): The number of clusters.
    iterations (int): The maximum number of iterations
    that should be performed.

    Returns:
    tuple: (C, clss)
        C is a numpy.ndarray of shape (k, d) containing the
        centroid means for each cluster.
        clss is a numpy.ndarray of shape (n,) containing the
        index of the cluster in C that each data point belongs to.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    Cs = initialize(X, k)
    clss = np.argmin(np.linalg.norm(X[:, np.newaxis] - Cs, axis=2), axis=1)
    for i in range(iterations):
        Cs_copy = Cs.copy()
        for i in range(len(Cs)):
            if len(X[clss == i] > 0):
                Cs_copy[i] = np.mean(X[clss == i], axis=0)
            else:
                Cs_copy[i] = initialize(X, 1)
        clss = np.argmin(np.linalg.norm(X[:, np.newaxis] - Cs_copy, axis=2),
                         axis=1)
        if np.array_equal(Cs, Cs_copy):
            break
        Cs = Cs_copy
    return Cs, clss
