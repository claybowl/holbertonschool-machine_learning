#!/usr/bin/env python3
"""3-optimum
tests for the optimum number of
clusters by variance
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Parameters:
    X (numpy.ndarray): The data set. Shape (n, d).
    kmin (int): The minimum number of
    clusters to check for (inclusive).
    kmax (int): The maximum number of
    clusters to check for (inclusive).
    iterations (int): The maximum number of
    iterations for K-means.

    Returns:
    tuple: (results, d_vars)
        results is a list containing the outputs
        of K-means for each cluster size.
        d_vars is a list containing the difference in variance
        from the smallest cluster size for each cluster size.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0 or X.shape[0] < kmin:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0 or X.shape[0] < kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Create empty lists
    results = []
    vars = []
    d_vars = []

    # Calculate kmeans and variance through range of kmin to kmax
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        vars.append(variance(X, C))

    # Calculate d_vars from the smallest cluster size 4 each cluster
    for var in vars:
        d_vars.append(vars[0] - var)

    return results, d_vars
