#!/usr/bin/env python3
"""Module 0-pca
Function that performs PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """Function that performs PCA on a dataset"""
    _, _, V = np.linalg.svd(X)
    cumulative = np.cumsum(V**2) / np.sum(V**2)
    r = (np.argwhere(cumulative >= var))[0, 0]
    return V.T[:, :r + 1]
