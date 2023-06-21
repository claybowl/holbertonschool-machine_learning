#!/usr/bin/env python3
"""module 0-markov_chain
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Function that determines the probability of a Markov chain being in a
    particular state after a specified number of iterations.

    Parameters:
    - P is a square 2D numpy.ndarray
    of shape (n, n) representing the transition
      matrix.
    - s is a numpy.ndarray of shape (1, n) representing the probability of
      starting in each state.
    - t is the number of iterations that the Markov chain has been through.

    Returns:
    - a numpy.ndarray of shape (1, n) representing
    the probability of being in a
      specific state after t iterations, or None on failure.
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

    # Check if s is a numpy.ndarray of shape (1, n)
    if type(s) is not np.ndarray or s.shape != (1, n):
        return None

    # Check if t is an integer and is greater than or equal to 0
    if type(t) is not int or t < 0:
        return None

    try:
        # Compute the state probabilities after t iterations
        state_probabilities = np.matmul(s, np.linalg.matrix_power(P, t))
        return state_probabilities
    except Exception:
        return None
