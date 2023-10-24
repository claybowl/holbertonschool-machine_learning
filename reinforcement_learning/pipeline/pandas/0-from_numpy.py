#!/usr/bin/env python3
"""module 0-from_numpy
Fucntion that creates a pd.Dataframe from a np.ndarray.
"""
import numpy as np
import pandas as pd


def from_numpy(array):
    df = pd.DataFrame(array)
    return df
