#!/usr/bin/env python3
""" 0x02. Keras """
import tensorflow.keras as k
import json


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.
    """
    model_config = network.get_config()
    with open(filename, 'w') as f:
        json.dump(model_config, f)


def load_config(filename):
    """
    Loads a model with a specific configuration.
    """
    with open(filename, 'r') as f:
        model_config = json.load(f)
    model = K.models.Model.from_config(model_config)
    return model
