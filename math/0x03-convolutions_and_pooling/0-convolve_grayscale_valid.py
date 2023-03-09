#!/usr/bin/env python3
"""return the output array
containing the convolved images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    # extract the dimensions of the images
    m, h, w = images.shape
    kh, kw = kernel.shape
    # compute output height and width based on kernel shape.
    output_h = h - kh + 1
    output_w = w - kw + 1
    # Create output array of zeros w/ appropriate shape.
    output = np.zeros((m, output_h, output_w))
    # Iterate over each pixel in output array.
	# For each pixel, extract the corresponding slice
	# of the input image and multiply it by the kernel.
	# Then, sum the result and store it in the output array.
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))
    return output
