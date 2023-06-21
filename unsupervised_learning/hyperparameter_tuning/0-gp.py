#!/usr/bin/env python3
"""module 0-gp.py
Creates class GaussianProcess that represents a
noiseless 1D Gaussian process
"""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        X_init: numpy.ndarray of shape (t, 1)
        representing the inputs already sampled
        Y_init: numpy.ndarray of shape (t, 1)
        representing the outputs for each input in X_init
        l: length parameter for the kernel
        sigma_f: standard deviation given to
        the output of the black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between
        two matrices using the RBF kernel."""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
