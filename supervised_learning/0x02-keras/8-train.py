#!/usr/bin/env python3
""" 0x02. Keras """
import tensorflow.keras as k


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """train_model: Trains a model using mini-batch gradient descent."""
    if validation_data:
        early_stopping_callback = None
        if early_stopping:
            early_stopping_callback = K.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience,
                restore_best_weights=True)
        learning_rate_decay_callback = None
        if learning_rate_decay:
            def schedule(epoch):
                return alpha / (1 + decay_rate * epoch)
            learning_rate_decay_callback = K.callbacks.LearningRateScheduler(
                schedule, verbose=1)
        save_best_callback = None
        if save_best:
            save_best_callback = K.callbacks.ModelCheckpoint(
                filepath, monitor='val_loss', save_best_only=True,
                save_weights_only=False, mode='min', verbose=1)
        callbacks = [early_stopping_callback,
                     learning_rate_decay_callback, save_best_callback]
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle, validation_data=validation_data,
                              callbacks=callbacks)
    else:
        history = network.fit(data, labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle)
    return history
