#!/usr/bin/env python3
""" 0x02. Keras """
import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i, (n, activation) in enumerate(zip(layers, activations)):
        x = K.layers.Dense(n, activation=activation,
                           kernel_regularizer=K.regularizers.l2(lambtha))(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
    outputs = x
    model = K.models.Model(inputs, outputs)
    return model
