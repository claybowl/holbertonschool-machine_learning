#!/usr/bin/env python3
""" 0x02. Keras """
import tensorflow.keras as k


def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):
    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          verbose=verbose, shuffle=shuffle)
    return history
