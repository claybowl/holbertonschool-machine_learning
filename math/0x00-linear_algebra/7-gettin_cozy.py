#!/usr/bin/env python3
"""consule 7-gettin_cozy
concatenates two matrices along a specific axis
"""
import numpy as np


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return
        else:
            return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return
        else:
            return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
