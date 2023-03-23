#!/usr/bin/env python3
"""2-identity_block.py
builds an identity block as described in
Deep Residual Learning for Image Recognition (2015)
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Function that builds an identity block"""
    F11, F3, F12 = filters

    # First 1x1 convolution, batch normalization, and ReLU activation
    x = K.layers.Conv2D(F11, (1, 1), padding='same',
                        kernel_initializer='he_normal')(A_prev)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation('relu')(x)

    # 3x3 convolution, batch normalization, and ReLU activation
    x = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer='he_normal')(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation('relu')(x)

    # Second 1x1 convolution, batch normalization
    x = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer='he_normal')(x)
    x = K.layers.BatchNormalization(axis=-1)(x)

    # Add the input tensor and apply ReLU activation
    x = K.layers.Add()([x, A_prev])
    x = K.layers.Activation('relu')(x)

    return x
