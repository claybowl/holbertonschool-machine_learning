#!/usr/bin/env python3
"""Module preprocess_data
Preprocesses the data for the model 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


# Load the datasets
bitmax_data = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
coinbase_data = pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')

# Combine the datasets
df = pd.concat([bitmax_data, coinbase_data])


# Remove rows with NaN values
df_clean = df.dropna()

# Turn Unixtime into pd timestamps for training
df_clean['DateTime'] = pd.to_datetime(df_clean['Timestamp'], unit='s')

# Drop the 'Volume_(BTC)' column
df_clean = df_clean.drop(columns=['Volume_(BTC)'])

# Select numerical features for scaling
features_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume_(Currency)', 'Weighted_Price']

# Initialize a scaler
scaler = StandardScaler()

# Fit the scaler to the data and transform
df_clean[features_to_scale] = scaler.fit_transform(df_clean[features_to_scale])

# Display the first few rows of the rescaled data
df_clean.head()


# Preprocess the data
data = df_clean['Close'].values
data = data.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

df_clean.to_csv('cleaned_and_rescaled_data.csv', index=False)
