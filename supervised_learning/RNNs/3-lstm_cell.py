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

    def forward(self, h_prev, x_t):
        """Function that performs forward propagation for one time step"""
        # h_prev is a numpy.ndarray of shape (m, h) containing the previous
        #
        concat = np.concatenate((h_prev, x_t), axis=1)
        # forget gate
        f_t = self.sigmoid(np.dot(concat, self.Wf) + self.bf)
        # update gate
        u_t = self.sigmoid(np.dot(concat, self.Wu) + self.bu)
        # intermediate cell state
        c_intermediate = np.tanh(np.dot(concat, self.Wc) + self.bc)
        # cell state
        c_next = f_t * c_prev + u_t * c_intermediate
        # output gate
        o_t = self.sigmoid(np.dot(concat, self.Wo) + self.bo)
        # hidden state
        h_next = o_t * np.tanh(c_next)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y
