#!/usr/bin/env python3
"""module forecast_btc
Forecasts the price of Bitcoin using a LSTM model
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
train, test = data[0:train_size, :], data[train_size:len(data), :]


def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


look_back = 24
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Convert the numpy arrays to tf.data.Dataset
train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
test_data = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

# Batch the datasets
train_data = train_data.batch(20)
test_data = test_data.batch(20)

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,
          input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=1, batch_size=1, verbose=2)

# Predicting future stock prices for next hour
inputs = data[len(data) - look_back:].reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(look_back, inputs.shape[0]):
    X_test.append(inputs[i-look_back:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
closing_price = model.predict(X_test)

# Plotting the results
plt.figure(figsize=(16, 8))
plt.plot(closing_price, label='Predicted Close Price')
plt.title('BTC Price Prediction')
plt.xlabel('Time')
plt.ylabel('BTC Price')
plt.legend(loc='best')
plt.show()
