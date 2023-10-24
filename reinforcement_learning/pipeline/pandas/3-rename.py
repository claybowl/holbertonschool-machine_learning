#!/usr/bin/env python3
"""module 3-rename
Script that renames data columns
"""
import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data using the provided function
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Rename the column Timestamp to Datetime
df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)

# Convert the 'Datetime' column from UNIX timestamp to human readable format
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

# Select only 'Datetime' and 'Close' columns
df = df[['Datetime', 'Close']]

print(df.tail())
