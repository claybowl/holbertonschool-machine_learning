#!/usr/bin/env python3
"""module 12-bracin_the_elements
Function that performs element-wise addition, subtraction
multiplication, and division.
"""


def np_elementwise(mat1, mat2):
    """Returns the elementwise operations"""
    element_sum = [[a + b for a, b in zip(row1, row2)]for row1, row2 in
                   zip(mat1, mat2)]
    element_sub = [[a - b for a, b in zip(row1, row2)]for row1, row2 in
                   zip(mat1, mat2)]
    element_mul = [[a * b for a, b in zip(row1, row2)]for row1, row2 in
                   zip(mat1, mat2)]
    element_div = [[a / b for a, b in zip(row1, row2)]for row1, row2 in
                   zip(mat1, mat2)]
    return (element_sum, element_sub, element_mul element_div)
