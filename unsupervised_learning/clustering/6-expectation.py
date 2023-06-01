#!/usr/bin/env python3
"""6-expectation
calculates the expectation step in the EM algorithm for a GMM
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.

    Parameters:
    X (numpy.ndarray): The data set. Shape (n, d).
    pi (numpy.ndarray): The priors for each cluster. Shape (k,).
    m (numpy.ndarray): The centroid means for each cluster. Shape (k, d).
    S (numpy.ndarray): The covariance matrices for
    each cluster. Shape (k, d, d).

    Returns:
    tuple: (g, l)
        g is a numpy.ndarray of shape (k, n) containing the
        posterior probabilities for each
        data point in each cluster.
        l is the total log likelihood.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    g = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        g[i] = pi[i] * P

    g_sum = np.sum(g, axis=0)
    g /= g_sum

    l = np.sum(np.log(g_sum))

    return g, l
