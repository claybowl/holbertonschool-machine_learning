#!/usr/bin/env python3
"""Performs pooling on images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
   """Performs pooling on images."""
   m, h, w, c = images.shape
   kh, kw = kernel_shape
   sh, sw = stride

   # Determine output shape
   out_h = int((h - kh) / sh) + 1
   out_w = int((w - kw) / sw) + 1

   # Initialize output array
   out = np.zeros((m, out_h, out_w, c))

   # Perform pooling
   for i in range(out_h):
       for j in range(out_w):
           img = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
           if mode == 'max':
               out[:, i, j, :] = np.max(img, axis=(1, 2))
           elif mode == 'avg':
               out[:, i, j, :] = np.mean(img, axis=(1, 2))

   return out
