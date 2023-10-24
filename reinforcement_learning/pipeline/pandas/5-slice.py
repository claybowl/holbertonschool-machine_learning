#!/usr/bin/env python3
"""module 5-slice
Slice the pd.Dataframe along the columns 'High', 'Low', 'Close'
and 'Volume_BTC', taking every 60th row.
"""
import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data using the provided function
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Select the columns 'High', 'Low', 'Close', and 'Volume_BTC'
# Take every 60th row
df = df[['High', 'Low', 'Close', 'Volume_BTC']][::60]

print(df.tail())
