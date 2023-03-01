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
    for l in range(L, 0, -1):
        dW = (1 / m) * np.matmul(dZ,
                                 cache["A" + str(l - 1)]
                                 .T)+ (lambtha / m) * weights["W" + str(l)]
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.matmul(weights["W" + str(l)].T, dZ)
        dZ = np.multiply(dA, np.int64(cache["A" + str(l - 1)] > 0))
        weights["W" + str(l)] = weights["W" + str(l)] - alpha * dW
        weights["b" + str(l)] = weights["b" + str(l)] - alpha * db
    return weights
