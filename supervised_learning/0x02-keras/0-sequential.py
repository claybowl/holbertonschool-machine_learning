#!/usr/bin/env python3
""" 0x02. Keras """
import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library"""
    model = k.models.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(k.layers.Dense(layers[i], activation=activations[i],
                                     input_shape=(nx,),
                                     kernel_regularizer=k.regularizers.l2(lambtha)))
        else:
            model.add(k.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=k.regularizers.l2(lambtha)))
        if i < len(layers) - 1:
            model.add(k.layers.Dropout(1 - keep_prob))
    return model
