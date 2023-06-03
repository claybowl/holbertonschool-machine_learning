#!/usr/bin/env python3
"""5-pdf
calculates the probability density function of a Gaussian distribution
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    Parameters:
    X (numpy.ndarray): The data points whose
    PDF should be evaluated. Shape (n, d).
    m (numpy.ndarray): The mean of the distribution. Shape (d,).
    S (numpy.ndarray): The covariance of the distribution.
    Shape (d, d).

    Returns:
    numpy.ndarray: The PDF values for each data point. Shape (n,).
    """
    _, dimensions = X.shape

    # Calculate the inverse and determinant of the covariance matrix
    inv_covariance = np.linalg.inv(S)
    det_covariance = np.linalg.det(S)

    if det_covariance <= 0:
        return None

    X_minus_mean = X - m

    # Calculate the exponent of the PDF formula
    exponent = -0.5 * np.sum(X_minus_mean @
                             inv_covariance * X_minus_mean, axis=1)
    # Calculate the coefficient of the PDF formula
    coeff = 1 / np.sqrt((2 * np.pi) ** dimensions * det_covariance)

    # Calculate the PDF values, and set mins to 1e-300
    pdf_values = coeff * np.exp(exponent)
    pdf_values = np.maximum(pdf_values, 1e-300)

    return pdf_values
