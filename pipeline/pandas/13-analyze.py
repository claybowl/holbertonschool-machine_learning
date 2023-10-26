#!/usr/bin/env python3
import pandas as pd

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')


# Remove the Timestamp column
# The `axis=1` parameter indicates we
# are dropping a column (not a row)
# If axis was 0, it would mean we're dropping a row
df = df.drop('Timestamp', axis=1)

# Calculate descriptive statistics
# The describe() function provides statistics like mean, count,
# std deviation, min, 25%, 50%, 75%, and max values for each column
stats = df.describe()

# Print the statistics
print(stats)
