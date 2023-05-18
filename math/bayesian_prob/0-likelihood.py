#!/usr/bin/env python3
"""module 0-likelihood.py
Contains the function likelihood
"""
import numpy as np


def likelihood(x, n, P):
    """ Calculates the likelihood of obtaining this data """
    if type(n) != int or n <= 0:
        raise ValueError("n must be a positive integer")

    if type(x) != int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) != np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculating factorial using numpy prod and arange
    fact_n = np.prod(np.arange(1, n + 1))
    fact_x = np.prod(np.arange(1, x + 1))
    fact_n_minus_x = np.prod(np.arange(1, n - x + 1))

    comb = fact_n / (fact_x * fact_n_minus_x)

    likelihood = comb * np.power(P, x) * np.power((1 - P), (n - x))

    return likelihood
