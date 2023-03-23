#!/usr/bin/env python3
"""7-densenet121.py
builds the DenseNet-121 architecture as described
in Densely Connected Convolutional Networks (2017)
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Function that builds the DenseNet-121 architecture"""
    input_shape = (224, 224, 3)

    # Define the input layer
    input_layer = K.Input(shape=input_shape)

    # Initial convolution layers
    x = K.layers.Conv2D(2 * growth_rate, (7, 7), strides=(2, 2),
                        padding='same',
                        kernel_initializer='he_normal')(input_layer)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Dense block 1 and transition layer 1
    x, nb_filters = dense_block(
        x, nb_filters=2 * growth_rate, growth_rate=growth_rate, layers=6)
    x, nb_filters = transition_layer(
        x, nb_filters=nb_filters, compression=compression)

    # Dense block 2 and transition layer 2
    x, nb_filters = dense_block(
        x, nb_filters=nb_filters, growth_rate=growth_rate, layers=12)
    x, nb_filters = transition_layer(
        x, nb_filters=nb_filters, compression=compression)

    # Dense block 3 and transition layer 3
    x, nb_filters = dense_block(
        x, nb_filters=nb_filters, growth_rate=growth_rate, layers=24)
    x, nb_filters = transition_layer(
        x, nb_filters=nb_filters, compression=compression)

    # Dense block 4
    x, nb_filters = dense_block(
        x, nb_filters=nb_filters, growth_rate=growth_rate, layers=16)

    # Final layers
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer='he_normal')(x)

    # Create the Keras model
    model = K.Model(inputs=input_layer, outputs=x)

    return model
