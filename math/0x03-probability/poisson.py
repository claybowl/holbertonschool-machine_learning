#!/usr/bin/env python3
"""Module poisson
Creates a class: Poisson
"""


class Poisson:
    """Class Poisson"""

    def __init__(self, data=None, lambtha=1.):
        """initializes Poisson class"""
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
        """Calculates the probablity mass function"""
        e = 2.7182818285
        k = int(k)
        if k <= 0:
            return 0
        kFact = 1
        for i in range(1, k + 1):
            kFact *= i
        return (self.lambtha ** k) * ((e ** -self.lambtha) / kFact)

    def cdf(self, k):
        """Calculates the Cumulative Distribution Function"""
        e = 2.7182818285
        k = int(k)
        if k <= 0:
            return 0
        cdf = 0
        for x in range(0, k + 1):
            xFact = 1
            for i in range(1, x + 1):
                xFact *= i
            cdf += (self.lambtha ** x) * ((e ** -self.lambtha) / xFact)
        return cdf

    def factorial(self, n):
        """Returns the factorial of a given number"""
        if n == 0 or n == 1:
            return 1
        else:
            return n * self.factorial(n-1)
