#!/usr/bin/env python3
"""module 0-rnn_cell
contains the class RNNCell
"""
import numpy as np


def rnn(rnn_cell, x,, h_0):
    """Function that performs forward propagation for a simple RNN"""
    t, m, i = x.shape
    _, h = h_0.shape
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))
    H[0] = h_0
    for i in range(t):
        h_next, y = rnn_cell.forward(H[i], x[i])
        H[i + 1] = h_next
        Y[i] = y
    return H, Y
