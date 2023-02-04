#!/usr/bin/env python3
"""module 3-neuron
Writes a class Neuron that defines a single neuron
performing binary classification.
"""
import numpy as np


class Neuron():
    """class Neuron"""

    def __init__(self, nx):
        """initializes a neuron with the given number of inputs"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter function for W"""
        return self.__W

    @property
    def b(self):
        """getter function for b"""
        return self.__b

    @property
    def A(self):
        """getter function for A"""
        return self.__A

    def forward_prop(self, X):
        """calculates the forward propagation of the neuron"""
        # Calculates linear activation.
        Z = np.matmul(self.__W, X) + self.__b
        # sigmoid activation function
        sigmoid = (1 / (1 + np.exp(-Z)))
        # Weighted sum Z passed through activation function
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """calculates the cost model using logistic regression"""
        # m is number of examples
        m = Y.shape[1]
        # formula for calculating cost
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
