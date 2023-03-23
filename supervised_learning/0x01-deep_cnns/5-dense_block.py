#!/usr/bin/env python3
"""5-dense_block.py
builds a dense block as described in
Densely Connected Convolutional Networks (2017)
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Function that builds a dense block"""
    for i in range(layers):
        # Batch normalization, ReLU activation, 1x1 convolution
        inter_channel = 4 * growth_rate
        bottleneck = K.layers.BatchNormalization(axis=-1)(X)
        bottleneck = K.layers.Activation('relu')(bottleneck)
        bottleneck = K.layers.Conv2D(
            inter_channel, (1, 1), padding='same',
            kernel_initializer='he_normal')(bottleneck)

        # Batch normalization, ReLU activation, 3x3 convolution
        conv_layer = K.layers.BatchNormalization(axis=-1)(bottleneck)
        conv_layer = K.layers.Activation('relu')(conv_layer)
        conv_layer = K.layers.Conv2D(
            growth_rate, (3, 3), padding='same',
            kernel_initializer='he_normal')(conv_layer)

        # Concatenate the output of the current layer with the previous layers
        X = K.layers.Concatenate(axis=-1)([X, conv_layer])

        # Update the number of filters
        nb_filters += growth_rate

    return X, nb_filters
