#!/usr/bin/env python3
"""module 0-rnn_cell
contains the class RNNCell
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Function to implement a bidirectional
    recurrent neural network (Bi-RNN).

    Parameters:
    bi_cell: An instance of the BidirectionalCell class.
    X: A numpy array of shape (t, m, i) containing the input data.
    h_0: A numpy array of shape (m, h) containing the initial
    hidden state in the forward direction.
    h_t: A numpy array of shape (m, h) containing the initial
    hidden state in the backward direction.

    Returns:
    H: A numpy array of shape (t, m, 2 * h) containing the
    concatenated hidden states from both directions.
    Y: A numpy array of shape (t, m, o)
    containing the outputs of the bidirectional RNN.
    """

    # Get the dimensions of the input data and
	# the initial hidden states
    t, m, i = X.shape
    _, h = h_0.shape

    # Get the number of outputs of the bidirectional cell
    o = bi_cell.Wy.shape[1]

    # Initialize the arrays to store the forward
	# and backward hidden states and the outputs
    Hf = np.zeros((t + 1, m, h))
    Hb = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    # Set the initial hidden states
    Hf[0] = h_0
    Hb[-1] = h_t

    # Loop over the time steps
    for step in range(t):
        # Compute the forward and backward hidden states
        Hf[step + 1] = bi_cell.forward(Hf[step], X[step])
        Hb[-step - 2] = bi_cell.backward(Hb[-step - 1], X[-step - 1])

    # Concatenate the forward and backward hidden states
    H = np.concatenate((Hf[1:], Hb[:-1]), axis=-1)

    # Compute the outputs
    Y = bi_cell.output(H)

    return H, Y
