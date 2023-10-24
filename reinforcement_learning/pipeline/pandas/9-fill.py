#!/usr/bin/env python3
"""module 9-fill
Fills in the missing data points in the pd.DataFrame
according to requirements.
"""
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')


# Remove Weighted_Price column
df = df.drop('Weighted_Price', axis=1)

# Fill missing values in Close with previous row value
df.Close.fillna(method='pad', inplace=True)

# Fill missing values in High, Low, Open with same row's Close value
df[['High', 'Low', 'Open']].fillna(df.Close, inplace=True)

# Fill missing values in Volume_(BTC) and Volume_(Currency) with 0
df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0, inplace=True)

print(df.head())
print(df.tail())
