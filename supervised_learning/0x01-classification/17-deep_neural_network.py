#!/usr/bin/env python3
"""module 17-deep_neural_network
Class Deep Neural Network
"""
import numpy as np


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        for i in range(len(layers)):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(1, self.__L + 1):
            if l == 1:
                self.__weights["W" + str(l)] = np.random.randn(
                    layers[l - 1], nx) * np.sqrt(2 / nx)
            else:
                self.__weights["W" + str(l)] = np.random.randn(
                    layers[l - 1], layers[l - 2]) * np.sqrt(2 / layers[l - 2])
            self.__weights["b" + str(l)] = np.zeros((layers[l - 1], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
