#!/usr/bin/env python3
""" 0x02. Keras """
import tensorflow.keras as k


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """train_model: Trains a model using mini-batch gradient descent."""

    if validation_data:
        early_stopping = early_stopping and (patience > 0)
        if early_stopping:
            early_stopper = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience)

        if learning_rate_decay:
            learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: alpha / (1 + decay_rate * epoch), verbose=1)

    if validation_data and (early_stopping or learning_rate_decay):
        callbacks = [early_stopper, learning_rate_scheduler] if early_stopping and learning_rate_decay else [
            early_stopper] if early_stopping else [learning_rate_scheduler]
    else:
        callbacks = None

    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose,
                          shuffle=shuffle, validation_data=validation_data, callbacks=callbacks)

    return history
