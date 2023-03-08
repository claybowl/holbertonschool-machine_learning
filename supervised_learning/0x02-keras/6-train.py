#!/usr/bin/env python3
""" 0x02. Keras """
import tensorflow.keras as k


def train_model(network, data, labels, batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.
    """
    if validation_data:
        history = network.fit(data, labels, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=shuffle,
                              validation_data=validation_data, callbacks=[early_stopping_callback(early_stopping, patience)])
    else:
        history = network.fit(data, labels, epochs=epochs,
                              batch_size=batch_size, verbose=verbose, shuffle=shuffle)
    return history


def early_stopping_callback(early_stopping, patience):
    """
    Returns a EarlyStopping callback to be used in model.fit
    """
    if early_stopping:
        return K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    return None
