#!/usr/bin/env python3
"""module 16-deep_neural_network
Class Deep Neural Network
"""
import numpy as np


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        """Initialize the neural network"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            he_init = np.sqrt(2 / nx)
            self.weights['W' +
                         str(l)] = np.random.randn(layers[l - 1], nx) * he_init
            self.weights['b' + str(l)] = np.zeros((layers[l - 1], 1))
            nx = layers[l - 1]
