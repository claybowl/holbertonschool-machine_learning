#!/usr/bin/env python3
"""module 3-bayes_opt.py
Creates the class BayesianOptimization that performs
Bayesian optimization on a noiseless 1d Gaussian
process.
"""
from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Creates the class BayesianOptimization"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Initialize Bayesian Optimization.

        f: the black-box function to be optimized.
        X_init: numpy.ndarray of shape (t, 1) representing
        the inputs already sampled with the black-box function.
        Y_init: numpy.ndarray of shape (t, 1) representing
        the outputs of the black-box function for each input in X_init.
        bounds: tuple of (min, max) representing the bounds
        of the space in which to look for the optimal point.
        ac_samples: number of samples that should be analyzed
        during acquisition.
        l: length parameter for the kernel.
        sigma_f: standard deviation given to
        the output of the black-box function.
        xsi: exploration-exploitation factor for acquisition.
        minimize: bool determining whether optimization should be
        performed for minimization (True) or maximization (False).
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """calculates the next best sample location using the
        Expected Improvement acquisition algorithm"""
        # Predict the mean and standard deviation for sample points
        mu, sigma = self.gp.predict(self.X_s)

        # Best objective function value observed
        f_plus = np.min(self.gp.Y) if self.minimize else np.max(self.gp.Y)

        # Calculate Z
        with np.errstate(divide='warn'):
            Z = (mu - f_plus - self.xsi) / sigma
            Z[sigma == 0.0] = 0.0

        # Calculate the Expected Improvement
        EI = (mu - f_plus - self.xsi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma == 0.0] = 0.0

        # Find the next best sample point
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
