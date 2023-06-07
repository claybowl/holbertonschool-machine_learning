#!/usr/bin/env python3
"""module 2-absorbing
"""
import numpy as np


def absorbing(P):
    """
    Function that determines if a Markov chain is absorbing.

    Parameters:
    - P is a is a square 2D numpy.ndarray of shape (n, n) representing the transition matrix.

    Returns:
    - True if the Markov chain is absorbing, or False on failure.
    """

    # Check if P is a square 2D numpy.ndarray
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    n, m = P.shape
    if n != m:
        return None

    # Check if every row of P sums up to 1
    if not np.isclose(np.sum(P, axis=1), 1).all():
        return None

    # Check if P is absorbing (there is at least one row with a 1 on its diagonal)
    if not np.any(np.diag(P) == 1):
        return False

    # Check if the states with a 1 on the diagonal are accessible from all other states
    for i in range(n):
        if P[i, i] == 1:
            for j in range(n):
                if i != j and np.all(np.linalg.matrix_power(P, n)[j, :] == 0):
                    return False

    return True
