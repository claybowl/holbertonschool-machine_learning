#!/usr/bin/env python3
"""module 1-gp.py
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
        if X1.ndim == 1:
            X1 = X1[:, None]
        if X2.ndim == 1:
            X2 = X2[:, None]
        assert X1.shape[1] == X2.shape[1]

        dist = np.sum(X1 ** 2, axis=1)[:, None] + \
            np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
        K = self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * dist)
        return K

    def predict(self, X_s):
        """
        Predict the mean and standard deviation of samples at X_s.
        X_s is a numpy.ndarray of shape (s, 1) containing all the points whose
        mean and standard deviation should be calculated
        s is the number of sample points
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s) + 1e-8 * np.eye(len(X_s))
        K_inv = np.linalg.inv(self.K)

        # Mean and covariance of posterior predictive distribution
        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s.flatten(), cov_s.flatten()
