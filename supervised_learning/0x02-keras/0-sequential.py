#!/usr/bin/env python3
""" 0x02. Keras """
import keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library
    """
    model = keras.Sequential()
    model.add(keras.layers.Dense(layers[0], activation=activations[0], input_shape=(nx,),
                                 kernel_regularizer=regularizers.l2(lambtha)))
    for i in range(1, len(layers)):
        model.add(keras.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=regularizers.l2(lambtha)))
        if i < len(layers) - 1:
            model.add(keras.layers.Dropout(1 - keep_prob))
    return model
