#!/usr/bin/env python3
"""module 1-l2_reg_gradient_descent.py
Write a function,
def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
that updates the weights and biases of a neural network using
gradient descent with L2 regularization:

Y is a one-hot numpy.ndarray of shape (classes, m)
that contains the correct labels for the data
classes is the number of classes
m is the number of data points
weights is a dictionary of the weights and biases of the neural network
cache is a dictionary of the outputs of each layer of the neural network
alpha is the learning rate
lambtha is the L2 regularization parameter
L is the number of layers of the network
The neural network uses tanh activations on each
layer except the last, which uses a softmax activation
The weights and biases of the network should be updated in place
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the weights and biases of neural network"""
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        dW = (1/m) * np.dot(dZ, cache['A' + str(i-1)
                                      ].T) + (lambtha/m) * weights[
                                      'W' + str(i)]
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(weights['W' + str(i)].T, dZ)
        if i > 1:
            dZ = dA * (1 - np.power(cache['A' + str(i-1)], 2))
        weights["W" + str(i)] = weights["W" + str(i)] - alpha * dW
        weights["b" + str(i)] = weights["b" + str(i)] - alpha * db
    return weights
