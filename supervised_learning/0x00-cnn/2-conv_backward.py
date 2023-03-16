#!/usr/bin/env python3
"""Function for convolutional back prop. It takes in 5
parameters:
dZ : numpy.ndarray of shape (m, h_new, w_new, c_new)
A_prev : numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
W : numpy.ndarray of shape (kh, kw, c_prev, c_new)
b : numpy.ndarray of shape (1, 1, 1, c_new)
padding : string that is either same or valid, indicating
the type of padding used
stride : tuple of (sh, sw)
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Function that performs back propagation
    over a convolutional layer"""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    elif padding == "valid":
        ph = 0
        pw = 0

    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    A_slice = A_prev_pad[i, h_start:h_end, w_start:w_end, :]
                    dZ_value = dZ[i, h, w, c]

                    dA_prev_pad[i, h_start:h_end,
                                w_start:w_end,
                                :] += W[:, :, :,
                                        c] * dZ_value[np.newaxis,
                                                      np.newaxis]
                    dW[:, :, :, c] += A_slice * dZ_value
                    db[:, :, :, c] += dZ_value

    if padding == "same":
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    elif padding == "valid":
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
