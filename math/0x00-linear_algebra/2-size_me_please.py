#!/usr/bin/env python3
"""Module 2-size_me_please
Calculates the shape of a matrix
"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape