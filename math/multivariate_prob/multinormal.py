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
