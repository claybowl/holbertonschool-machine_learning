#!/usr/bin/env python3
""" Pooling Forward Prop """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """function that performs forward propagation over a pooling layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    A_new = np.zeros((m, h_new, w_new, c_prev))

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_prev):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    A_slice = A_prev[i, h_start:h_end, w_start:w_end, c]

                    if mode == 'max':
                        A_new[i, h, w, c] = np.max(A_slice)
                    elif mode == 'avg':
                        A_new[i, h, w, c] = np.mean(A_slice)

    return A_new
