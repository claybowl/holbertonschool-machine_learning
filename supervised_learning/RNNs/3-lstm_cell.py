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
        """sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """Function that performs forward propagation for one time step"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        f_t = self.sigmoid(np.dot(concat, self.Wf) + self.bf)
        u_t = self.sigmoid(np.dot(concat, self.Wu) + self.bu)
        c_intermediate = np.tanh(np.dot(concat, self.Wc) + self.bc)
        c_next = f_t * c_prev + u_t * c_intermediate
        o_t = self.sigmoid(np.dot(concat, self.Wo) + self.bo)
        h_next = o_t * np.tanh(c_next)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, c_next, y
