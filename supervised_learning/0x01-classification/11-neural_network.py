#!/usr/bin/env python3
"""module 11-neural_network
Defines a neural network with one hidden layer performing binary classification
"""
import numpy as np


class NeuralNetwork:
    """class NeuralNetwork"""

    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def sigmoid(self, Z):
        """sigmoid function"""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """calculates the forward propagation"""
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(Z1)
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(Z2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        cost = -(np.matmul(Y, np.log(A).T) + np.matmul(1 -
                 Y, np.log(1.0000001 - A).T)) / Y.shape[1]
        cost = np.squeeze(cost)
        return cost
