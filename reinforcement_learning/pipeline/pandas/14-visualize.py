#!/usr/bin/env python3
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')


# Remove the Weighted_Price column
df = df.drop('Weighted_Price', axis=1)

# Rename the Timestamp column to Date
df.rename(columns={'Timestamp': 'Date'}, inplace=True)

# Convert the timestamp values to date values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index the data frame on Date
df.set_index('Date', inplace=True)

# Missing values in Close should be set to the previous row value
df['Close'].fillna(method='ffill', inplace=True)

# Missing values in High, Low, Open should be set to the same row’s Close value
df[['High', 'Low', 'Open']].fillna(method='bfill', inplace=True)

# Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0, inplace=True)

# Group the values of the same day for plotting
day_groups = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
}).rename(columns={'Date': 'day'})

# Plot the data from 2017 and beyond at daily intervals
plt.figure(figsize=(12, 6))
plt.plot(day_groups['day'], day_groups['High'], label='High')
plt.plot(day_groups['day'], day_groups['Low'], label='Low')
plt.plot(day_groups['day'], day_groups['Close'], label='Close')
plt.plot(day_groups['day'], day_groups['Volume_(BTC)'], label='Volume (BTC)')
plt.plot(day_groups['day'], day_groups['Volume_(Currency)'], label='Volume (Currency)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Daily Prices and Volumes, 2017-2019')
plt.legend()
plt.show()
