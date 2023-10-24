#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')


# Set the Timestamp column as the index for both DataFrames
df1 = df1.set_index('Timestamp')
df2 = df2.set_index('Timestamp')

# Concatenate the two DataFrames
df = pd.concat([df1, df2], axis=0)

# Add keys to the data
df['key'] = 'coinbase'
df.loc[df['Timestamp'] >= 1417411920, 'key'] = 'bitstamp'

print(df)
