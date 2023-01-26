#!/usr/bin/env python3
"""Module poisson
Creates a class: Poisson
"""
import math


class Poisson:
    """Class Poisson"""
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data)/len(data))

    def pmf(self, k):
        k = int(k)
        if k < 0:
            return 0
        return (self.lambtha**k * math.exp(-self.lambtha)) / math.factorial(k)

    def cdf(self, k):
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k+1):
            cdf += (self.lambtha**i * math.exp(-self.lambtha)) / math.factorial(i)
        return cdf
