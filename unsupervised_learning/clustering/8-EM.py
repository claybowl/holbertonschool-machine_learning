#!/usr/bin/env python3
"""8-EM
performs the expectation maximization for a GMM
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM.

    Parameters:
    X (numpy.ndarray): The data set. Shape (n, d).
    k (int): The number of clusters.
    iterations (int): The maximum number of iterations for the algorithm.
    tol (float): The tolerance of the log likelihood.
    verbose (bool): Whether to print information about the algorithm.

    Returns:
    tuple: The final values of pi, m, S, g, and ll.
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    # Initialize pi, m, and S
    pi, m, S = initialize(X, k)

    # Initialize log likelihood
    log_likelihood = [0, 0]

    # Perform expectation maximization
    for i in range(iterations + 1):
        # Expectation step
        g, log_likelihood[1] = expectation(X, pi, m, S)
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}"
                  .format(i, log_likelihood[1].round(5)))
        if abs(log_likelihood[0] - log_likelihood[1]) <= tol:
            if verbose:
                print("Log Likelihood after {} iterations: {}"
                      .format(i, log_likelihood[1].round(5)))
            break
        if i < iterations:
            pi, m, S = maximization(X, g)

        # Update log likelihood
        log_likelihood[0] = log_likelihood[1]

    return pi, m, S, g, log_likelihood[1]
