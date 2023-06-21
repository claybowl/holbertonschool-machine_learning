#!/usr/bin/env python3
"""module 1-correlation.py
Calculates a correlation matrix
"""
import numpy as np


def correlation(C):
    """Calculates a correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]
    std_dev = np.sqrt(np.diag(C))
    outer_std_dev = np.outer(std_dev, std_dev)
    correlation = C / outer_std_dev

    return correlation
