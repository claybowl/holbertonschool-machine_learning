#!/usr/bin/env python3
"""9-BIC
finds the best number of clusters for a GMM
using the Bayesian Information Criterion:
"""
import numpy as np


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):

    # Initialize variables
    if kmax is None:
        kmax = X.shape[0]
    l = np.empty(kmax - kmin + 1)
    b = np.empty_like(l)
    best_result = None
    best_k = None
    best_bic = float('-inf')

    # Perform EM algorithm for each k
    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        p = k * (X.shape[1] * (X.shape[1] + 1) / 2 + X.shape[1] + 1)
        bic = p * np.log(X.shape[0]) - 2 * log_likelihood
        l[k - kmin] = log_likelihood
        b[k - kmin] = bic
        if bic > best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, l, b
