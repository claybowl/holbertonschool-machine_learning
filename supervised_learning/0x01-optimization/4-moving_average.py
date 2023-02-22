#!/usr/bin/env python3
"""module 4-moving_average
This function calculates the moving average
of a given array of data.
"""
import numpy as np


def moving_average(data, window_size):
    """The function returns an array of the moving averages"""
    moving_averages = []
    prev = 0
    for i in range(len(data)):
        if i == 0:
            prev = data[i]
        else:
            curr = window_size * data[i] + (1 - window_size) * prev
            moving_average.append(curr)
            prev = curr

    return moving_averages
