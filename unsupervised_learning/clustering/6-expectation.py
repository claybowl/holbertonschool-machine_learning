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
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None
    if X.shape[1] != m.shape[1] or m.shape[1] != S.shape[1]:
        return None, None
    if S.shape[1] != S.shape[2]:
        return None, None
    if pi.shape[0] != m.shape[0] or m.shape[0] != S.shape[0]:
        return None, None
    if False in [np.isclose(pi.sum(), 1)]:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    g = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        g[i] = pi[i] * P

    g_sum = np.sum(g, axis=0)
    g /= g_sum

    r = np.sum(np.log(g_sum))

    return g, r
