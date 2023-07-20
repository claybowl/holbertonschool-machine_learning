#!/usr/bin/env python3
"""module 0-rnn_cell
contains the class RNNCell
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Function to implement a deep RNN.

    Parameters:
    rnn_cells: list of RNNCell instances of length
    l that will be used for the forward prop
    X: data to be used, given as a numpy.ndarray of shape (t, m, i)
    h_0: initial hidden state, given as a numpy.ndarray of shape (l, m, h)

    Returns:
    H: numpy.ndarray containing all of the hidden states
    Y: numpy.ndarray containing all of the outputs
    """

    # Extract dimensions from h_0 and X
    l, m, h = h_0.shape  # l: layers, m: batch size, h: hidden units
    t, _, i = X.shape  # t: time steps, i: input size

    # Get output size from the last rnn cell
    o = rnn_cells[-1].Wy.shape[1]  # o: output size

    # Initialize H and Y
    H = np.zeros((t + 1, l, m, h))  # Hidden states
    Y = np.zeros((t, m, o))  # Outputs

    # Set initial hidden state
    H[0] = h_0

    # Loop over time steps
    for step in range(t):
        h_prev = X[step]  # Current input

        # Loop over layers
        for layer in range(l):
            # Forward prop through this layer
            h_next, y = rnn_cells[layer].forward(H[step, layer], h_prev)

            # Save hidden state for this layer and step
            H[step + 1, layer] = h_next

            # The output of this layer is the input of the next
            h_prev = h_next

        # Save output for this step
        Y[step] = y

    return H, Y
