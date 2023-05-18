#!/usr/bin/env python3
"""module 5-definiteness.py
Calculates the definitiveness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) is not 2 or matrix.shape[0] is not matrix.shape[1]:
        return None

    # Check if matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    # Calculate eigenvalues
    eigenvalues = np.linalg. eigvals(matrix)

    # Check the definiteness
    if all(eigenvalue > 0 for eigenvalue in eigenvalues):
        return "Positive definite"
    elif all(eigenvalue >= 0 for eigenvalue in eigenvalues):
        return "Positive semi-definite"
    elif all(eigenvalue < 0 for eigenvalue in eigenvalues):
        return "Negative definite"
    elif all(eigenvalue <= 0 for eigenvalue in eigenvalues):
        return "Negative semi-definite"
    else:
        return "Indefinite"

    return None
