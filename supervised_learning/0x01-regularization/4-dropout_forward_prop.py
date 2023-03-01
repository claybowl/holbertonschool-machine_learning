#!/usr/bin/env python3
"""module 4-dropout_forward_prop.py
Write a function def dropout_forward_prop(X, weights, L, keep_prob):
that conducts forward propagation using Dropout:
X is a numpy.ndarray of shape (nx, m)
containing the input data for the network
nx is the number of input features
m is the number of data points
weights is a dictionary of the weights and biases of the neural network
L the number of layers in the network
keep_prob is the probability that a node will be kept
All layers except the last should use the tanh activation function
The last layer should use the softmax activation function
Returns: a dictionary containing the outputs of each layer and
the dropout mask used on each layer (see example for format)
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout"""
    cache = {}
    A = X
    for l in range(1, L):
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]
        Z = np.dot(W, A) + b
        A = np.tanh(Z)
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < keep_prob).astype(int)
        A = np.multiply(A, D)
        A = A / keep_prob
        cache['D' + str(l)] = D
        cache['A' + str(l)] = A
    W = weights['W' + str(L)]
    b = weights['b' + str(L)]
    Z = np.dot(W, A) + b
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    cache['A' + str(L)] = A
    return cache
