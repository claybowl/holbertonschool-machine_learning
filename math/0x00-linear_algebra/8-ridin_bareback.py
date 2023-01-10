#!/usr/bin/env python3
"""module 8-ridin_bareback
Function that performs matrix multiplication.
"""
import numpy as np


def mat_mul(mat1, mat2):
    """matrix multiplication"""
    product = np.dot(mat1, mat2)
    return product
