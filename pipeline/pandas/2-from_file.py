#!/usr/bin/env python3
"""module 2-from_file
Imports data from file as pd.DataFrame.
"""
import pandas as pd

def from_file(filename, delimiter):
    """
    Load data from a file into a pandas DataFrame.

    Parameters:
    - filename: str
        The path to the file to be read.
    - delimiter: str
        The column separator for the file.

    Returns:
    - df: pd.DataFrame
        The loaded pandas DataFrame.
    """
    # Read file using Pandas and return the resulting Dataframe
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
