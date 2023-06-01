#!/usr/bin/env python3
"""8-EM
performs the expectation maximization for a GMM
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000,
                             tol=1e-5, verbose=False):

    # Initialize variables
    pi, m, S = initialize(X, k)
    g, _ = expectation(X, pi, m, S)
    l_prev = 0

    # Perform EM algorithm
    for i in range(iterations):
        pi, m, S = maximization(X, g)
        g, l = expectation(X, pi, m, S)

        # Check for verbose
        if verbose and (i % 10 == 0 or i == iterations - 1):
            print('Log Likelihood after {} iterations:
                  {}'.format(i, round(l, 5)))

        # Check for early stopping
        if abs(l - l_prev) <= tol:
            break
        l_prev = l

    return pi, m, S, g, l
