#!/usr/bin/env python3
"""module 16-deep_neural_network
Class Deep Neural Network
"""
import numpy as np


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or not layers:
            raise TypeError("layers must be a list of positive integers")
        for layer in layers:
            if type(layer) is not int or layer < 1:
                raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(self.L):
            if l == 0:
                self.weights["W" + str(l + 1)
                             ] = np.random.randn(layers[l], nx) * np.sqrt(2/nx)
                self.weights["b" + str(l + 1)] = np.zeros((layers[l], 1))
            else:
                self.weights["W" + str(l + 1)] = np.random.randn(layers[l],
                                                                 layers[l - 1]) * np.sqrt(2/layers[l - 1])
                self.weights["b" + str(l + 1)] = np.zeros((layers[l], 1))
