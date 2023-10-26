#!/usr/bin/env python3
"""module 7-high
Sorts the pd.Dataframe by the 'High' price in descending order.
"""
import pandas as pd

from_file = __import__('2-from_file').from_file

# use the sort_values function and specify descending order
df = df.sort_values(by='High', ascending=False)

print(df.head())
