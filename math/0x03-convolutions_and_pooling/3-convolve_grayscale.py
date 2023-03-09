#!/usr/bin/env python3
"""Performs a convolution on grayscale
images.
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph = max((h - 1) * sh + kh - h, 0) // 2
        pw = max((w - 1) * sw + kw - w, 0) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    images_padded = np.pad(images, pad_width=(
        (0, 0), (ph, ph), (pw, pw)), mode='constant')
    conv_h = (h + 2 * ph - kh) // sh + 1
    conv_w = (w + 2 * pw - kw) // sw + 1
    convolved_images = np.zeros((m, conv_h, conv_w))

    for i in range(conv_h):
        for j in range(conv_w):
            x = i * sh
            y = j * sw
            images_slide = images_padded[:, x:x+kh, y:y+kw]
            element_wise = np.multiply(images_slide, kernel)
            convolved_images[:, i, j] = np.sum(element_wise, axis=(1, 2))

    return convolved_images
