#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')


# Remove the Timestamp column
df = df.drop('Timestamp', axis=1)

# Calculate descriptive statistics
stats = df.describe()

print(stats)
