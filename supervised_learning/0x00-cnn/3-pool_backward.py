#!/usr/bin/env python3
"""Function that performs back propagation
over a pooling layer of a neural network:"""


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """function that performs back propagation"""
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    if mode == 'max':
                        mask = (A_prev[i, h_start:h_end,
                                       w_start:w_end, ch] == np.max(
                            A_prev[i, h_start:h_end,
                                   w_start:w_end, ch]))
                        dA_prev[i, h_start:h_end, w_start:w_end,
                                ch] += mask * dA[i, h, w, ch]
                    elif mode == 'avg':
                        avg_value = dA[i, h, w, ch] / (kh * kw)
                        dA_prev[i, h_start:h_end, w_start:w_end,
                                ch] += np.ones((kh, kw)) * avg_value

    return dA_prev
