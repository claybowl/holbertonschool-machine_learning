#!/usr/bin/env python3
"""3-projection_block.py
builds a projection block as described in
Deep Residual Learning for Image Recognition (2015)
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """builds a projection block"""
    F11, F3, F12 = filters

    # First 1x1 convolution, batch normalization, and ReLU activation
    x = K.layers.Conv2D(F11, (1, 1), strides=(
        s, s), padding='same', kernel_initializer='he_normal')(A_prev)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation('relu')(x)

    # 3x3 convolution, batch normalization, and ReLU activation
    x = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer='he_normal')(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation('relu')(x)

    # Second 1x1 convolution, batch normalization in the main path
    x = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer='he_normal')(x)
    x = K.layers.BatchNormalization(axis=-1)(x)

    # 1x1 convolution and batch normalization in the shortcut connection
    shortcut = K.layers.Conv2D(F12, (1, 1), strides=(
        s, s), padding='same', kernel_initializer='he_normal')(A_prev)
    shortcut = K.layers.BatchNormalization(axis=-1)(shortcut)

    # Add the main path and shortcut connection and apply ReLU activation
    x = K.layers.Add()([x, shortcut])
    x = K.layers.Activation('relu')(x)

    return x
