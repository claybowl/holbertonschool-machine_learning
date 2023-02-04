#!/usr/bin/env python3
"""module 13-cats_got_your_tongue
Function that concatenates two matrices along a specific axis.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenates two matrices"""
    return np.concatenate((mat1, mat2), axis=axis)
