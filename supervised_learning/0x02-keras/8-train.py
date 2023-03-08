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
        validation_data_ = (validation_data[0], validation_data[1])
    else:
        validation_data_ = None

    if early_stopping and validation_data:
        early_stopping_callback = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)
        callbacks = [early_stopping_callback]
    else:
        callbacks = []

    if learning_rate_decay and validation_data:
        def learning_rate_scheduler(epoch):
            return alpha / (1 + decay_rate * epoch)

        learning_rate_callback = K.callbacks.LearningRateScheduler(
            learning_rate_scheduler, verbose=1)
        callbacks.append(learning_rate_callback)

    if save_best and validation_data:
        best_model_callback = K.callbacks.ModelCheckpoint(
            filepath, save_best_only=True,
            monitor='val_loss', mode='min')
        callbacks.append(best_model_callback)

    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose, shuffle=shuffle,
                          validation_data=validation_data_,
                          callbacks=callbacks)

    return history
