#!/usr/bin/env python3
"""
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    # obtain shape of images and kernel
    m, h, w = images.shape
    kh, kw = kernel.shape

    # calculate padding
    ph = int((kh - 1) / 2)
    pw = int((kw - 1) / 2)

    # pad the images with zeros
    images_padded = np.pad(images, pad_width=(
        (0,), (ph,), (pw,)), mode='constant')

    # initialize output
    output = np.zeros((m, h, w))

    # perform convolution
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(
                images_padded[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return output
