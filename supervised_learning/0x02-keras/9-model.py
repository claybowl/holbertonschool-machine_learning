#!/usr/bin/env python3
""" 0x02. Keras """
import tensorflow.keras as k


def save_model(network, filename):
    """Saves an entire model.

    Arguments:
    network -- the model to save
    filename -- the path of the file that the model should be saved to

    Returns:
    None
    """
    network.save(filename)
    return None


def load_model(filename):
    """Loads an entire model.

    Arguments:
    filename -- the path of the file that the model should be loaded from

    Returns:
    the loaded model
    """
    return K.models.load_model(filename)
