#!/usr/bin/env python3
"""module 4-viterbi
"""
import numpy as np


def viterbi(O, E, T, V, s):
    """
    Function that calculates the most likely
    sequence of hidden states for a hidden Markov model.

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
    - path, P, or None, None on failure
        - path is a list of length T that
        contains the most likely sequence of hidden states.
        - P is the probability of obtaining the path sequence.
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

    # Initialize the Viterbi path probabilities and the backpointers
    V_path = np.zeros((N, T_))
    V_path[:, 0] = s * E[:, O[0]]
    backpointers = np.zeros((N, T_), dtype=int)

    # Compute the Viterbi path probabilities and the backpointers
    for t in range(1, T_):
        for j in range(N):
            prob = V_path[:, t - 1] * T[:, j] * E[j, O[t]]
            V_path[j, t] = np.max(prob)
            backpointers[j, t] = np.argmax(prob)

    # Compute the most likely sequence of hidden states
    path = [np.argmax(V_path[:, -1])]
    for t in range(T_ - 1, 0, -1):
        path.append(backpointers[path[-1], t])
    path = path[::-1]

    # Compute the probability of obtaining the path sequence
    P = np.max(V_path[:, -1])

    return path, P
