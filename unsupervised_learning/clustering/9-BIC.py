#!/usr/bin/env python3
"""9-BIC
finds the best number of clusters for a GMM
using the Bayesian Information Criterion:
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion.

    Parameters:
    X (numpy.ndarray): The data set. Shape (n, d).
    kmin (int): The minimum number of clusters to check for (inclusive).
    kmax (int): The maximum number of clusters to check for (inclusive).
    iterations (int): The maximum number of iterations for the EM algorithm.
    tol (float): The tolerance for the EM algorithm.
    verbose (bool): Whether to print information about the algorithm.

    Returns:
    tuple: The best number of clusters, the best result, the log likelihoods,
    and the BIC values.
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if kmin >= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    # Initialize variables
    n, d = X.shape
    bic_values = []
    log_likelihoods = []
    results = []
    cluster_counts = []

    # Loop over all possible number of clusters
    for k in range(kmin, kmax + 1):
        # Perform expectation maximization
        pi, m, S, _, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)

        # Store results
        results.append((pi, m, S))
        cluster_counts.append(k)
        log_likelihoods.append(log_likelihood)

        # Calculate BIC
        p = k * d * (d + 1) / 2 + d * k + k - 1
        bic = p * np.log(n) - 2 * log_likelihood
        bic_values.append(bic)

    # Convert to numpy arrays
    bic_values = np.array(bic_values)

    # Find the best number of clusters
    best_k_index = np.argmin(bic_values)

    return (cluster_counts[best_k_index], results[best_k_index],
            log_likelihoods, bic_values)
