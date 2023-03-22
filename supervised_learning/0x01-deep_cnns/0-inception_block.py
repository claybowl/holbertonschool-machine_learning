#!/usr/bin/env python3
"""0-inception_block.py
builds an inception block as described
in Going Deeper with Convolutions (2014)
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution
    conv1x1 = K.layers.Conv2D(
        F1, (1, 1), padding='same', activation='relu')(A_prev)

    # 1x1 convolution before the 3x3 convolution
    conv3x3_reduce = K.layers.Conv2D(
        F3R, (1, 1), padding='same', activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(
        F3, (3, 3), padding='same', activation='relu')(conv3x3_reduce)

    # 1x1 convolution before the 5x5 convolution
    conv5x5_reduce = K.layers.Conv2D(
        F5R, (1, 1), padding='same', activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(
        F5, (5, 5), padding='same', activation='relu')(conv5x5_reduce)

    # Max pooling and 1x1 convolution after the max pooling
    max_pool = K.layers.MaxPooling2D(
        (3, 3), strides=(1, 1), padding='same')(A_prev)
    conv_pool = K.layers.Conv2D(
        FPP, (1, 1), padding='same', activation='relu')(max_pool)

    # Concatenate the outputs
    inception_output = K.layers.concatenate(
        [conv1x1, conv3x3, conv5x5, conv_pool], axis=-1)

    return inception_output
