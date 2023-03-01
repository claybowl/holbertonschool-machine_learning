#!/usr/bin/env python3
"""module 5-dropout_gradient_descent.py
Write a function def dropout_gradient_descent(Y, weights, cache,
alpha, keep_prob, L): that updates the weights of a
neural network with Dropout regularization using gradient descent:

Y is a one-hot numpy.ndarray of shape (classes, m)
that contains the correct labels for the data
classes is the number of classes
m is the number of data points
weights is a dictionary of the weights and biases of the neural network
cache is a dictionary of the outputs and dropout
masks of each layer of the neural network
alpha is the learning rate
keep_prob is the probability that a node will be kept
L is the number of layers of the network
All layers use the tanh activation function except the last,
which uses the softmax activation function
The weights of the network should be updated in place
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates the weights of a neural network with Dropout regularization"""
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    for l in range(L, 0, -1):
        A = cache['A' + str(l - 1)]
        D = cache['D' + str(l - 1)]
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]
        dW = (1 / m) * np.dot(dZ, A.T) + (lambda / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(W.T, dZ)
        dA = np.multiply(dA, D)
        dA = dA / keep_prob
        dZ = dA * (1 - np.power(A, 2))
        weights['W' + str(l)] = W - alpha * dW
        weights['b' + str(l)] = b - alpha * db
