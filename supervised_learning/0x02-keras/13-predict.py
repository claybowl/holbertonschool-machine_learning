#!/usr/bin/env python3
""" 0x02. Keras """
import tensorflow.keras as k


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network."""
    prediction = network.predict(data)
    if verbose:
        print("Prediction: ", prediction)
    return prediction
