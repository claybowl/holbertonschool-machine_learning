#!/usr/bin/env python3
"""module 3-flip_me_over
Returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """returns the transpose of a 2D matrix"""
    transpose = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
    return transpose
