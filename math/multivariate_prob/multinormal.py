#!/usr/bin/env python3
"""module MultiNormal.py
Contains the class MultiNormal
"""
import numpy as np


class MultiNormal:
    """Class for Multivariate Normal distribution"""

    def __init__(self, data):
        """Class initializer"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        deviation = data - self.mean
        self.cov = np.matmul(deviation, deviation.T) / (n - 1)

    def pdf(self, x):
        """Calculates the PDF at a data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        denom = np.sqrt(((2 * np.pi) ** d) * det)
        exponent = np.exp(-0.5 * np.dot(np.dot((x - self.mean).T, inv),
                                        (x - self.mean)))

        return float(exponent / denom)
