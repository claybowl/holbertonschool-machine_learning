#!/usr/bin/env python3
"""Performs a convolution on images using
multiple kernels.
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
   """Performs a convolution on images using multiple kernels"""
   m, h, w, c = images.shape
   kh, kw, _, nc = kernels.shape
   sh, sw = stride

   # Determine padding
   if padding == 'same':
       ph = int(((h - 1) * sh + kh - h) / 2) + 1
       pw = int(((w - 1) * sw + kw - w) / 2) + 1
   elif padding == 'valid':
       ph, pw = 0, 0
   else:
       ph, pw = padding

   # Add padding to images
   images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                          mode='constant', constant_values=0)

   # Calculate output dimensions
   oh = int((h + 2 * ph - kh) / sh) + 1
   ow = int((w + 2 * pw - kw) / sw) + 1

   # Initialize output array
   output = np.zeros((m, oh, ow, nc))

   # Perform convolution
   for i in range(oh):
       for j in range(ow):
           for k in range(nc):
               output[:, i, j, k] = np.sum(
                   images_padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :] *
                   kernels[:, :, :, k], axis=(1, 2, 3))
