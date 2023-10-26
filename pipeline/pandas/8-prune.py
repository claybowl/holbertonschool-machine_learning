#!/usr/bin/env python3
"""module 8-prune
Removes entries in the pd.DataFrame where 'Close' is NaN.
"""
import pandas as pd

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Drop rows where 'Close' is NaN
df = df.dropna(subset=['Close'])

print(df.head())
