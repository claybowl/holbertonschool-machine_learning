#!/usr/bin/env python3
"""7-maximization
calculates the expectation step in the EM algorithm for a GMM
"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.

    Parameters:
    X (numpy.ndarray): The data set. Shape (n, d).
    g (numpy.ndarray): The posterior probabilities for each data point in each cluster. Shape (k, n).

    Returns:
    tuple: (pi, m, S)
        pi is a numpy.ndarray of shape (k,) containing the updated priors for each cluster.
        m is a numpy.ndarray of shape (k, d) containing the updated centroid means for each cluster.
        S is a numpy.ndarray of shape (k, d, d) containing the updated covariance matrices for each cluster.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    pi = np.sum(g, axis=1) / n

    m = np.dot(g, X) / np.sum(g, axis=1, keepdims=True)

    S = np.zeros((k, d, d))
    for i in range(k):
        X_m = X - m[i]
        S[i] = np.dot(g[i] * X_m.T, X_m) / np.sum(g[i])

    return pi, m, S
