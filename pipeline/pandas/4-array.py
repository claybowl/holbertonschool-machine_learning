#!/usr/bin/env python3
"""module 4-array
Take last 10 rows of columns 'High' and 'Close' 
and convert them into numpy.ndarray.
"""
import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data using the provided function
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Select the last 10 rows of the columns 'High' and 'Close'
subset = df[['High', 'Close']].tail(10)

# Convert the DataFrame slice into a numpy array
A = subset.values

print(A)
