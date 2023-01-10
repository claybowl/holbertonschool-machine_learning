#!/usr/bin/env python3
"""module 8-ridin_bareback
Function that performs matrix multiplication.
"""
import numpy as np


def mat_mul(mat1, mat2):
    """matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return
    else:
        product = []
        for i in range(len(mat1)):
            row = []
            for j in range(len(mat2[0])):
                element = 0
                for k in range(len(mat1[0])):
                    element += mat1[i][k] * mat2[k][j]
                row.append(element)
            product.append(row)
        return product
