#!/usr/bin/env python3
"""module 16-deep_neural_network
Class Deep Neural Network
"""
import numpy as np


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if any(not isinstance(i, int) or i <= 0 for i in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for l in range(1, self.L + 1):
            w = np.random.randn(layers[l - 1], nx) * np.sqrt(2 / nx)
            b = np.zeros((layers[l - 1], 1))
            self.weights["W" + str(l)] = w
            self.weights["b" + str(l)] = b
            nx = layers[l - 1]
