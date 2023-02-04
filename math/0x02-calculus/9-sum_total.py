#!/usr/bin/env python3
"""module 9-sum_total
calculates the summation of i^2"""


def summation_i_squared(n):
    """ Calculates the summation of i^2"""

    if n < 1:
        return None

    return sum(map(lambda sq: sq**2, list(range(1, n + 1))))
