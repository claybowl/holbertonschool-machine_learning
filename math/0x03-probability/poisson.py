#!/usr/bin/env python3
"""Module poisson
Creates a class: Poisson
"""


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
		"""pmf"""
        ## convert k to int if it's not already
        k = int(k)
        ## if k is out of range, return 0
        if k < 0:
            return 0
        ## calculate and return the PMF value for k
        return (self.lambtha**k * 2.718281828**(-self.lambtha)) /
		        self.factorial(k)

    def cdf(self, k):
        """cdf"""
        ## convert k to int if it's not already
        k = int(k)
        ## if k is out of range, return 0
        if k < 0:
            return 0
        ## calculate and return the CDF value for k
        cdf = 0
        for i in range(k+1):
            cdf += (self.lambtha**i * 2.718281828**(-self.lambtha)) /
			        self.factorial(i)
        return cdf

    def factorial(self,n):
        """factorial"""
        if n == 0 or n == 1:
            return 1
        else:
            return n * self.factorial(n-1)
