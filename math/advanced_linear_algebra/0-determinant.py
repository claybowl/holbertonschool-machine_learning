#!/usr/bin/env python3
"""0-detrminant.py
Calculates the determinant of a matrix
"""


def determinant(matrix):
    """Calculates the determinant of a matrix"""
    # Check if matrix is list of lists
    matrix_size = len(matrix)
    if not all(len(row) == matrix_size for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case for 0x0 matrix
    if matrix_size == 0:
        return 1

    # Base case for 1x1 matrix
    if matrix_size == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if matrix_size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive calculation of the determinant for matrices larger than 2x2
    det = 0
    for index, value in enumerate(matrix[0]):
        # Create a submatrix by removing the first row and the current column
        submatrix = [row[:index] + row[index + 1:] for row in matrix[1:]]

        # Add or subtract the current value times the determinant of the submatrix
        det += value * determinant(submatrix) * (-1) ** index

    return det
