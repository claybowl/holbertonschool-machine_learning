#!/usr/bin/env python3
"""module 13-batch_norm
Normalizes an un-activated output of a neural
network using batch normalization.
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """implements the tensorflow with the 'staircase' parameter
    **Z** : un-activated output of a neural network
    **gamma** : parameter for scaling
    **beta** : parameter for shifting
    **epsilon** : small number for numerical stabilization
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    Z_tilde = gamma * Z_norm + beta

    # returns normalized and shifted un-activated output
    return Z_tilde
