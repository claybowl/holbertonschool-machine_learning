#!/usr/bin/env python3
"""Function that creates a pd.DataFrame from
a np.ndarray.
"""
import pandas as pd
import numpy as np


def from_numpy(array):
    """Create a DataFrame from a np.ndarray"""
    df = pd.DataFrame(array)
    return df
