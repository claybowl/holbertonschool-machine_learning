#!/usr/bin/env python3
""" that performs a convolution on
images with channels.
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """function that performs a convolution on"""
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    images_padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    out_h = int((h + 2 * ph - kh) / sh) + 1
    out_w = int((w + 2 * pw - kw) / sw) + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            img_slice = images_padded[:, i * sh:i *
                                      sh + kh, j * sw:j * sw + kw, :]
            output[:, i, j] = np.sum(img_slice * kernel, axis=(1, 2, 3))

    return output
