#!/usr/bin/env python3
"""module 0-rnn_cell
contains the class RNNCell
"""
import numpy as np


class BidirectionalCell:
    """
    Class that represents a bidirectional cell for a bidirectional RNN.
    """

    def __init__(self, i, h, o):
        """
        Initialize the cell.

        Parameters:
        i: integer, representing the dimensionality of the data
        h: integer, representing the dimensionality of the hidden state
        o: integer, representing the dimensionality of the outputs
        """
        # Weight matrices
        self.Whf = np.random.normal(size=(i + h, h))  # Forward hidden state
        self.Whb = np.random.normal(size=(i + h, h))  # Backward hidden state
        self.Wy = np.random.normal(size=(2 * h, o))  # Output

        # Bias vectors
        self.bhf = np.zeros((1, h))  # Forward hidden state
        self.bhb = np.zeros((1, h))  # Backward hidden state
        self.by = np.zeros((1, o))  # Output

    def softmax(self, x):
        """
        Compute softmax values for each sets of scores in x.

        Parameters:
        x: A numpy array which we want to apply softmax to.

        Returns:
        A numpy array representing softmax of input x.
        """
        e_x = np.exp(
            x - np.max(x))  # Subtract max(x) to compute
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Parameters:
        h_prev: numpy array of shape (m, h), contains the previous hidden state
        x_t: numpy array of shape (m, i), contains the data input for the cell

        Returns:
        h_next: next hidden state
        """
        # Concatenate h_prev and x_t to
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state using the formula: tanh(W . concat + b)
        h_next = np.tanh(np.dot(concat, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Perform backward propagation for one time step.

        Parameters:
        h_next: numpy array of shape (m, h), contains the next hidden state
        x_t: numpy array of shape (m, i), contains the data input for the cell

        Returns:
        h_prev: previous hidden state
        """
        # Concatenate h_next and x_t to match dimensions for matrix multiplication
        concat = np.concatenate((h_next, x_t), axis=1)

        # Compute previous hidden state using the formula: tanh(W . concat + b)
        h_prev = np.tanh(np.dot(concat, self.Whb) + self.bhb)

        return h_prev
