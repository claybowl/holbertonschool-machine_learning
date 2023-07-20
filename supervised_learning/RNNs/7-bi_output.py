#!/usr/bin/env python3
"""module 0-rnn_cell
contains the class RNNCell
"""
import numpy as np


class BidirectionalCell():
    """
    Bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden states
            o: dimensionality of the outputs

        Creates public instance attributes Whf, Whb, Wy, bhf, bhb, by
            Whf, bhf: hidden weights and biases in forward direction
            Whb, Bhb: hidden weights and biases in backward direction
            Wy, by: for the outputs
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2*h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for ONE time step
        Args:
            h_prev: np.ndarray shape(m, h) containing the previous hidden state
            x_t: np.ndarray shape(m, i) contains the data input for the cell
                m: batch size for the data

        Returns:
            h_next: the next hidden state
        """
        # Previous hidden layer and input data are what we put in
        cell_input = np.concatenate((h_prev, x_t), axis=1)

        # Next hidden state is tanh of (cell_input * weights + bias)
        h_next = np.tanh(np.matmul(cell_input, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Perform forward propagation for ONE time step in the backward direction
        Args:
            h_next: np.ndarray shape(m, h) containing the next hidden state
            x_t: np.ndarray shape(m, i) contains the data input for the cell
                m: batch size for the data

        Returns:
            h_prev: the next hidden state
        """
        # Next hidden layer and input data are what we put in
        cell_input = np.concatenate((h_next, x_t), axis=1)

        # Previous hidden state is tanh of (cell_input * weights + bias)
        h_prev = np.tanh(np.matmul(cell_input, self.Whb) + self.bhb)

        return h_prev

    def output(self, H):
        """
        Calculates all outputs for the RNN

        Args:
            H: np.ndarray shape(t, m, 2h) that contains the concatenated hidden
                states from both directions, excluding their initialized states
                t: number of time steps
                m: batch size for the data
                h: dimensionality of the hidden states

        Returns:
            Y: the outputs of the network
        """
        Y = np.dot(H, self.Wy) + self.by
        Y = np.exp(Y) / np.sum(np.exp(Y), axis=2, keepdims=True)

        return Y
