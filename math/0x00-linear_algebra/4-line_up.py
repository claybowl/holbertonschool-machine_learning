#!/usr/bin/env python3
"""module 4-line_up
Adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """adds two arrays of elements"""
    if len(arr1) != len(arr2):
        return
    else:
        sum_arr = [a + b for a, b in zip(arr1, arr2)]
        return sum_arr
