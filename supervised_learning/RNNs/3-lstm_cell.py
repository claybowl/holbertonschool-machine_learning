#!/usr/bin/env python3
"""module 0-rnn_cell
contains the class RNNCell
"""
import numpy as np


class LSTMCell:
    """class RNNCell"""

    def __init__(self, i, h, o):
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """softmax function"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """Function that performs forward propagation for one time step"""
        cell_input = np.concatenate((h_prev, x_t), axis=1)

        # Forget Gate
        f = self.sigmoid(np.dot(cell_input, self.Wf) + self.bf)
        # Update Gate
        u = self.sigmoid(np.dot(cell_input, self.Wu) + self.bu)
        # Output Gate
        o = self.sigmoid(np.dot(cell_input, self.Wo) + self.bo)

        # Intermediate Cell State
        c_intermediate = np.tanh(np.dot(cell_input, self.Wc) + self.bc)

        # Next Cell
        c_next = c_prev * f + u * c_intermediate

        # Next Hidden State
        h_next = o * np.tanh(c_next)

        # Compute Output
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, c_next, y
