#!/usr/bin/env python3
"""module 12-bracin_the_elements
Function that performs element-wise addition, subtraction
multiplication, and division.
"""
import numpy as np


def np_elementwise(mat1, mat2):
    """Returns the elementwise operations"""
    element_sum = np.add(mat1, mat2)
    element_sub = np.subtract(mat1, mat2)
    element_mul = np.multiply(mat1, mat2)
    element_div = np.divide(mat1, mat2)
    return (element_sum, element_sub, element_mul element_div)
