#!/usr/bin/env python3
"""module binomial
"""
e = 2.7182818285
pi = 3.1415926536


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Represents a binomial distribution"""
        self.n = int(n)
        self.p = float(p)
        self.data = data
        if data is None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("n must be a positive integer")
            if not 0 < p < 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            p = sum(data) / len(data)
            n = round(len(data) / p)
            self.n = n
            self.p = p

    def pmf(self, k):
        """Calculates the PMF for a given number of successes"""

    def cdf(self, k):
        """Calculates the CDF for a given number of successes"""
