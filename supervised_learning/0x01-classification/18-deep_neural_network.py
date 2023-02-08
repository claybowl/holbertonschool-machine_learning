#!/usr/bin/env python3
"""module 18-deep_neural_network
Class Deep Neural Network
"""
import numpy as np


class DeepNeuralNetwork:
    """Class Deep Neural Network"""

    def __init__(self, nx, layers):
        """initializes the neural network"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(1, self.L + 1):
            he_init = np.sqrt(2 / nx)
            self.weights['W' +
                         str(l)] = np.random.randn(layers[l - 1], nx) * he_init
            self.weights['b' + str(l)] = np.zeros((layers[l - 1], 1))
            nx = layers[l - 1]

    def forward_prop(self, X):
        """forward propagation"""
        self.__cache["A0"] = X
        for l in range(1, self.__L + 1):
            W = self.__weights["W" + str(l)]
            b = self.__weights["b" + str(l)]
            A_prev = self.__cache["A" + str(l - 1)]
            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache["A" + str(l)] = A

        return A, self.__cache

    @property
    def L(self):
        """getter for the number of layers"""
        return self.__L

    @property
    def cache(self):
        """getter for the cache"""
        return self.__cache

    @property
    def weights(self):
        """getter for the weights"""
        return self.__weights
