#!/usr/bin/env python3
"""module 6-baum_welch
"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Function that performs the Baum-Welch algorithm for
    a hidden Markov model.

    Parameters:
    - Observations is a numpy.ndarray of shape (T,) that
    contains the index of the observation.
    - Transition is a numpy.ndarray of shape (M, M) that
    contains the initialized transition probabilities.
    - Emission is a numpy.ndarray of shape (M, N) that
    contains the initialized emission probabilities.
    - Initial is a numpy.ndarray of shape (M, 1) that
    contains the initialized starting probabilities.
    - iterations is the number of times expectation-maximization
    should be performed.

    Returns:
    - the converged Transition, Emission, or None, None on failure
    """

    # Check if the inputs are valid
    if type(Observations) is not np.ndarray or len(Observations.shape) != 1:
        return None, None
    T = Observations.shape[0]
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    M, _ = Transition.shape
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    _, N = Emission.shape
    if type(Initial) is not np.ndarray or Initial.shape != (M, 1):
        return None, None

    # Initialize the updated emission probabilities and
	# transition probabilities
    Emission_new = np.copy(Emission)
    Transition_new = np.copy(Transition)

    # Perform the Baum-Welch algorithm
    for _ in range(iterations):
        Emission_old = np.copy(Emission_new)
        Transition_old = np.copy(Transition_new)

        # Compute the forward path probabilities
		# and the backward path probabilities
        alpha = np.zeros((M, T))
        alpha[:, 0] = Initial[:, 0] *
        Emission[:, Observations[0]]

        for t in range(1, T):
            for j in range(M):
                alpha[j, t] = alpha[:, t - 1].dot(Transition[:, j]) *
                Emission[j, Observations[t]]

        beta = np.zeros((M, T))
        beta[:, -1] = 1

        for t in range(T - 2, -1, -1):
            for i in range(M):
                beta[i, t] = (beta[:, t + 1] *
                              Emission[:,Observations[t + 1]]).dot(Transition[i, :])

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[:, t].T,Transition) *
                                 Emission[:, Observations[t + 1]].T, beta[:, t + 1])
            for i in range(M):
                numerator = alpha[i, t] * Transition[i, :] *
                Emission[:, Observations[t + 1]].T * beta[:, t + 1].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        Transition_new = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        denominator = np.sum(gamma, axis=1)
        for l in range(N):
            Emission_new[:, l] = np.sum(gamma[:, Observations == l], axis=1)

        Emission_new = np.divide(Emission_new, denominator.reshape((-1, 1)))

        if np.all(Emission_old ==
                  Emission_new) and np.all(Transition_old ==
                                           Transition_new):
            return Transition_new, Emission_new

    return Transition_new, Emission_new