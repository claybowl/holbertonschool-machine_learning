#!/usr/bin/env python3
"""6-transition_layer.py
builds a transition layer as described in
Densely Connected Convolutional Networks (2017)
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Function that builds a transition layer"""
    # Batch normalization, ReLU activation
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation('relu')(X)

    # Apply compression to the number of filters
    nb_filters = int(nb_filters * compression)

    # 1x1 convolution
    X = K.layers.Conv2D(nb_filters, (1, 1), padding='same',
                        kernel_initializer='he_normal')(X)

    # 2x2 average pooling
    X = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(X)

    return X, nb_filters
