#!/usr/bin/env python3
"""Performs forward propagation over a convolutional layer of
neural network.
A_prev : output from previous layer.
W : weights for the convolution.
b : bias for the convolution.
activation : activation function for the convolution.
 """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer of a neural"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    else:  # padding == "valid"
        ph, pw = 0, 0

    h_new = int((h_prev + 2 * ph - kh) / sh) + 1
    w_new = int((w_prev + 2 * pw - kw) / sw) + 1

    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph),
                                    (pw, pw), (0, 0)), mode='constant')
    Z = np.zeros((m, h_new, w_new, c_new))

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                i_start, i_end = i * sh, i * sh + kh
                j_start, j_end = j * sw, j * sw + kw
                A_slice = A_prev_padded[:, i_start:i_end, j_start:j_end, :]
                Z[:, i, j, k] = np.sum(A_slice * W[:, :, :, k],
                                       axis=(1, 2, 3)) + b[:, :, :, k]

    A = activation(Z)
    return A
