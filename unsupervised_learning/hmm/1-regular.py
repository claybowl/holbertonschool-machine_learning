#!/usr/bin/env python3
import numpy as np
"""module 1-regular
"""


def regular(P):
    """
    Function that determines the steady state probabilities
    of a regular Markov chain.

    Parameters:
    - P is a is a square 2D numpy.ndarray of shape
    (n, n) representing the transition matrix.

    Returns:
    - a numpy.ndarray of shape (1, n) representing
    the steady state probabilities, or None on failure.
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

    # Check if P is regular (all rows can be raised to a
    # power where all rows are positive)
    if not np.any(np.linalg.matrix_power(P, n) > 0):
        return None

    # Compute the steady state probabilities
    w, v = np.linalg.eig(P.T)
    steady_state = v[:, np.isclose(w, 1)]
    steady_state = steady_state[:, 0]
    steady_state = steady_state / np.sum(steady_state)

    return steady_state.reshape(1, n)
