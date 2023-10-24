#!/usr/bin/env python3
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')


# Drop the Weighted_Price column
df.drop(columns='Weighted_Price', inplace=True)

# Rename the Timestamp column to Date
df.rename(columns={'Timestamp': 'Date'}, inplace=True)\
# Convert the timestamp values to date values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Set the DataFrame index to the Date column
df.set_index('Date', inplace=True)

# Missing values in Close should be set to the previous row value
df['Close'].fillna(method='ffill', inplace=True)

# Missing values in High, Low, Open should be set to the same row’s Close value
df[['High', 'Low', 'Open']].fillna(df['Close'], inplace=True)

# Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0, inplace=True)

# Create boolean mask to filter dates during and after 2017 only.
df[df.index.year >= 2017]

# Group samples by criteria(day) and assign each grouping
# an aggregate operation.
daily_df = df.resample('D').agg({
    'High': 'max',
    'Low':'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)':'sum',
    'Volume_(Currency)': 'sum'
})

# Group the values of the same day for plotting
day_groups = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
}).rename(columns={'Date': 'day'})

# Visualize the data
daily_df[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']].plot(figsize=(14, 8))
plt.title("Bitcoin Data from 2017 Onwards")
plt.ylabel("Value")
plt.xlabel("Date")
plt.tight_layout()
plt.legend(loc='upper left')
plt.show()
