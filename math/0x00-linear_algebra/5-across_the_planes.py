#!/usr/bin/env python3
"""module 5-across_the_planes
Adds two matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """Adds two matrices element-wise"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return
    else:
        matrix_sum = [[a + b for a, b in zip(row1, row2)] for
                      row1, row2 in zip(mat1, mat2)]
        return matrix_sum
