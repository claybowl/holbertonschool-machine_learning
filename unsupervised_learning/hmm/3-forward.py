#!/usr/bin/env python3
"""module 3-forward
"""
import numpy as np


def forward(O, E, T, V, s):
    """
    Function that performs the forward
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
    - P, F, or None, None on failure
        - P is the likelihood of the
        observations given the model.
        - F is a numpy.ndarray of shape (N, T) that
        contains the forward path probabilities.
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

    # Initialize the forward path probabilities
    F = np.zeros((N, T_))
    F[:, 0] = s * E[:, O[0]]

    # Compute the forward path probabilities
    for t in range(1, T_):
        for j in range(N):
            F[j, t] = np.sum(F[:, t - 1] * T[:, j] * E[j, O[t]])

    # Compute the likelihood of the observations
    P = np.sum(F[:, -1])

    return P, F
