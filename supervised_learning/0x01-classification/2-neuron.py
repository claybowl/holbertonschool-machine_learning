#!/usr/bin/env python3
"""module 2-neuron
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

    # def sigmoid(self, x):
    #     """sigmoid activation function"""
    #     return 1/(1 + np.exp(-x))

    def forward_prop(self, X):
        """calculates the forward propagation of the neuron"""
        # Calculates linear activation.
        Z = np.matmul(self.__W, X) + self.__b
        # # sigmoid activation function
        sigmoid = (1 / (1 + np.exp(-Z)))
        # Weighted sum Z passed through activation function
        self.__A = sigmoid
        return self.__A
