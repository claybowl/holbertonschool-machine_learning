#!/usr/bin/env python3
"""module 5-backward
"""
import numpy as np


def backward(O, E, T, V, s):
    """
    Function that performs the backward
    algorithm for a hidden Markov model.

    Parameters:
    - O is a numpy.ndarray of shape (T,) that
    contains the index of the observation.
    - E is a numpy.ndarray of shape (N, M) that
    contains the emission probability of a specific
    observation given a hidden state.
    - T is a numpy.ndarray of shape (N, N) that
    contains the transition probabilities.
    - V is a numpy.ndarray of shape (N,) that
    contains the initial state probabilities.
    - s is a numpy.ndarray of shape (N,) that
    contains the initial state probabilities.

    Returns:
    - P, B, or None, None on failure
        - P is the likelihood of the observations
        given the model.
        - B is a numpy.ndarray of shape (N, T) that
        contains the backward path probabilities.
    """

    # Check if the inputs are valid
    if type(O) is not np.ndarray or len(O.shape) != 1:
        return None, None
    T_, = O.shape
    if type(E) is not np.ndarray or len(E.shape) != 2:
        return None, None
    N, M = E.shape
    if type(T) is not np.ndarray or T.shape != (N, N):
        return None, None
    if type(V) is not np.ndarray or V.shape != (N,):
        return None, None
    if type(s) is not np.ndarray or s.shape != (N,):
        return None, None

    # Initialize the backward path probabilities
    B = np.zeros((N, T_))
    B[:, -1] = np.ones((N,))

    # Compute the backward path probabilities
    for t in range(T_ - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(B[:, t + 1] * T[i, :] * E[:, O[t + 1]])

    # Compute the likelihood of the observations
    P = np.sum(s * E[:, O[0]] * B[:, 0])

    return P, B
