#!/usr/bin/env python3
"""module 3-adjugate.py
Calculates the adjugate matrix of a matrix
"""


def determinant(matrix):
    """Calculates the determinant of a matrix"""

    # Check if matrix is list of lists
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    for i in range(len(matrix)):
        if type(matrix[i]) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a square matrix")

    matrix_size = len(matrix)

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

        # Add or subtract current value times determinant of the submatrix
        det += value * determinant(submatrix) * (-1) ** index

    return det


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix"""
    # Checks if our matrix is a list of list
    if type(matrix) is not list or not all(isinstance(row,
                                                      list) for
                                           row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Checks if matrix is a square
    size = len(matrix)
    if not all(len(row) == size for row in matrix) or size == 0:
        raise ValueError("matrix must be non-empty square matrix")

    # Calculate the cofactor matrix
    cofactor_matrix = []
    for i in range(size):
        cofactor_row = []
        for j in range(size):
            sub_matrix = [row[:j] + row[j + 1:] for
                          row in (matrix[:i] +
                                  matrix[i + 1:])]
            cofactor_row.append(((-1) ** (i + j))
                                * determinant(sub_matrix))
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix


def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix"""

    # Check if matrix is a list of lists
    if type(matrix) is not list or not all(isinstance
                                           (row, list) for row
                                           in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    size = len(matrix)
    if not all(len(row) == size for row in matrix) or size == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    # If matrix is 1x1, adjugate is same matrix
    if size == 1:
        return matrix

    # Calculate the cofactor of the matrix
    cofactor_matrix = cofactor(matrix)

    # Transpose the cofactor matrix to get adjugate
    adjugate_matrix = list(map(list, zip(*cofactor_matrix)))

    return adjugate_matrix
