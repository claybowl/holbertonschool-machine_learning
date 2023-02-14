#!/usr/bin/env python3
"""module 1-create_layer.py
Initializes layer and activation
function.
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates layer and activation function"""
    # Implements 'He et. al' initializations for the layer weights
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )
    # Specifies number of nodes, activation function and initializer
    layer = tf.layers.dense(
        prev,
        n,
        activation=activation,
        kernel_initializer=initializer
    )
    # Applies the layer to the input 'prev'
    return layer
