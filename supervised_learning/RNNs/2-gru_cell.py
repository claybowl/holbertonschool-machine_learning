#!/usr/bin/env python3
"""module 0-rnn_cell
contains the class RNNCell
"""
import numpy as np


class GRUCell:
    """class RNNCell"""
    def __init__(self, i, h, o):
        """Constructor"""
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """softmax function"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """Function that performs forward propagation for one time step"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        z = self.sigmoid(np.dot(concat, self.Wz) + self.bz)
        r = self.sigmoid(np.dot(concat, self.Wr) + self.br)
        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_intermediate = np.tanh(np.dot(concat_r, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h_intermediate
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, y
