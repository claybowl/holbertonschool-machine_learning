#!/usr/bin/env python3
"""2-variance
Calculates the total intra-dcluster variance
for a data set
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set.

    Parameters:
    X (numpy.ndarray): The data set. Shape (n, d).
    C (numpy.ndarray): The centroid means for each cluster. Shape (k, d).

    Returns:
    float: The total variance.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    dist = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=-1))
    min_dist = np.min(dist, axis=0)

    var = np.sum(min_dist ** 2)

    return var
