#!/usr/bin/env python3
"""Performs a valid convolution on grayscale
images with custom padding.
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a valid convolution on grayscale"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad the images
    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)),
        mode='constant', constant_values=0)

    # Create output array
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1
    output = np.zeros((m, output_h, output_w))

    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = (
                kernel * padded_images[:, i:i+kh, j:j+kw]).sum(axis=(1, 2))

    return output
