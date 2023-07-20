#!/usr/bin/env python3
"""module 0-rnn_cell
contains the class RNNCell
"""
import numpy as np
import math


class RNNCell:

    def __init__(self, i, h, o):
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, 0))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
            concat = np.concatenate((h_prev, x_t), axis=1)
            h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)
            y = self.softmax(np.dot(h_next, self.Wy) + self.by)
            return h_next, y
