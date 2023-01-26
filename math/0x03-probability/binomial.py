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
            if n < 1:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum([(value - mean) ** 2 for value in data]) / len(data)
            self.p = 1 - (variance / mean)
            self.n = round(mean / self.p)
            self.p = mean / self.n

    def pmf(self, k):
        """Calculates the PMF for a given number of successes"""
        k = int(k)
        if k < 0:
            return 0

        def factorial(x):
            """factorials!"""
            if x < 0:
                raise ValueError("x must be >= 0")
            if x <= 1:
                return 1
            return x * factorial(x - 1)

        binCoef = factorial(self.n)/(factorial(k) * factorial(self.n - k))
        return binCoef * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the CDF for a given number of successes"""
        k = int(k)
        if k < 0:
            return 0
        return sum([self.pmf(i) for i in range(0, k + 1)])
