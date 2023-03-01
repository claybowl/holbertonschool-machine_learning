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
    for i in range(L, 0, -1):
        A_cur_layer = cache['A' + str(i)]
        if i == L:
            dZ = A_cur_layer - Y
        else:
            dZ = np.dot(weights['W' + str(i + 1)].T, dZ) * \
                (1 - np.power(A_cur_layer, 2))
            dZ = dZ * cache['D' + str(i)]
            dZ = dZ / keep_prob
        dW = np.dot(dZ, cache['A' + str(i - 1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        weights['W' + str(i)] = weights['W' + str(i)] - alpha * dW
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db
