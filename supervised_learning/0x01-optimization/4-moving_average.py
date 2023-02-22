#!/usr/bin/env python3
"""module 4-moving_average
This function calculates the moving average
 of a given array of data.
"""
import numpy as np
import tensorflow as tf


def moving_average(data, window_size):
    """The function returns an array of the moving averages"""
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        window_sum = 0
        for j in range(window_size):
            window_sum += data[i + j]
        moving_averages.append(window_sum / window_size)
    return moving_averages


data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
window_size = 3

print(moving_average(data, window_size))
