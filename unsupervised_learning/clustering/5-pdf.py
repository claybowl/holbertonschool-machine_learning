#!/usr/bin/env python3
"""5-pdf
calculates the probability density function of a Gaussian distribution
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    Parameters:
    X (numpy.ndarray): The data points whose PDF should be evaluated. Shape (n, d).
    m (numpy.ndarray): The mean of the distribution. Shape (d,).
    S (numpy.ndarray): The covariance of the distribution. Shape (d, d).

    Returns:
    numpy.ndarray: The PDF values for each data point. Shape (n,).
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape

    X_m = X - m

    S_inv = np.linalg.inv(S)

    fac = np.einsum('...k,kl,...l->...', X_m, S_inv, X_m)

    P = 1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(S))) * np.exp(-fac / 2)

    P = np.maximum(P, 1e-300)

    return P
