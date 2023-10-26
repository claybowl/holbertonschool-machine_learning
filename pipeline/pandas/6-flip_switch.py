#!/usr/bin/env python3
"""module 6-flip_switch
contains the function flip_switch.
"""
import pandas as pd

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
# df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Transpose the DataFrame
df = df.T

# Sort the DataFrame in reverse chronological order based on the 'Timestamp' values
df = df.sort_values(by='Timestamp', ascending=False)

print(df.tail(8))
