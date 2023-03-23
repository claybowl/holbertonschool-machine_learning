#!/usr/bin/env python3
"""4-resnet50.py
builds the ResNet-50 architecture as described in
Deep Residual Learning for Image Recognition (2015)
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Function that builds the ResNet-50 architecture"""
    input_shape = (224, 224, 3)

    # Define the input layer
    input_layer = K.Input(shape=input_shape)

    # Initial convolution layers
    x = K.layers.Conv2D(64, (7, 7), strides=(
        2, 2), padding='same', kernel_initializer='he_normal')(input_layer)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # First set of residual blocks
    x = projection_block(x, filters=(64, 64, 256), s=1)
    x = identity_block(x, filters=(64, 64, 256))
    x = identity_block(x, filters=(64, 64, 256))

    # Second set of residual blocks
    x = projection_block(x, filters=(128, 128, 512))
    x = identity_block(x, filters=(128, 128, 512))
    x = identity_block(x, filters=(128, 128, 512))
    x = identity_block(x, filters=(128, 128, 512))

    # Third set of residual blocks
    x = projection_block(x, filters=(256, 256, 1024))
    x = identity_block(x, filters=(256, 256, 1024))
    x = identity_block(x, filters=(256, 256, 1024))
    x = identity_block(x, filters=(256, 256, 1024))
    x = identity_block(x, filters=(256, 256, 1024))
    x = identity_block(x, filters=(256, 256, 1024))

    # Fourth set of residual blocks
    x = projection_block(x, filters=(512, 512, 2048))
    x = identity_block(x, filters=(512, 512, 2048))
    x = identity_block(x, filters=(512, 512, 2048))

    # Final layers
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer='he_normal')(x)

    # Create the Keras model
    model = K.Model(inputs=input_layer, outputs=x)

    return model
