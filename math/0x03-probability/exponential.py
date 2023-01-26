#!/usr/bin/env python3
"""module exponential
initalizes the exponential
"""


class Exponential:
    """A exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """initialize the exponential"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """calculates the pdf of time period"""
        if x <= 0:
            return 0
        return format(self.lambtha * 2.7182818285**(-self.lambtha*x), '.6f')

    def cdf(self, x):
        """Calculates the CDF for time peroid"""
        if x <= 0:
            return 0
        return format(1 - 2.7182818285**(-self.lambtha*x), '.6f')
