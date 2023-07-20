#!/usr/bin/env python3
"""module 0-rnn_cell
contains the class RNNCell
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Function to perform forward propagation for a bidirectional RNN.

    Parameters:
    bi_cell: an instance of the class BidirectionalCell
    X: numpy array of shape (t, m, i) containing the data input for the cell
        t: the maximum number of time steps
        m: the batch size
        i: the dimensionality of the data
    h_0: numpy array of shape (m, h) containing the initial hidden state in the forward direction
    h_t: numpy array of shape (m, h) containing the initial hidden state in the backward direction

    Returns:
    H: numpy array of shape (t, m, 2 * h) containing the concatenated hidden states from both directions, excluding their initialized states
    Y: numpy array of shape (t, m, o) containing the outputs of the bidirectional RNN
    """
    # Get the dimensions of the input data and the hidden state
    t, m, i = X.shape
    _, h = h_0.shape

    # Get the dimensionality of the output
    o = bi_cell.Wy.shape[1]

    # Initialize the forward and backward hidden states
    Hf = np.zeros((t + 1, m, h))  # Forward hidden states
    Hb = np.zeros((t + 1, m, h))  # Backward hidden states

    # Initialize the output
    Y = np.zeros((t, m, o))

    # Set the initial hidden states
    Hf[0] = h_0
    Hb[-1] = h_t

    # Loop over all time steps
    for step in range(t):
        # Forward propagation in forward direction
        Hf[step + 1] = bi_cell.forward(Hf[step], X[step])

        # Forward propagation in backward direction
        Hb[-step - 2] = bi_cell.backward(Hb[-step - 1], X[-step - 1])

    # Concatenate the forward and backward hidden states
    H = np.concatenate((Hf[1:], Hb[:-1]), axis=-1)

    # Compute the output
    Y = bi_cell.softmax(np.dot(H, bi_cell.Wy) + bi_cell.by)

    return H, Y

def output(self, H):
    """Output"""
    Y = np.dot(H, self.Wy) + self.by
    Y = np.exp(Y) / np.sum(np.exp(Y), axis=2, keepdims=True)

    return Y
