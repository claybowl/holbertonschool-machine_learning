#!/usr/bin/env python3
"""module 3-l2_reg_create_layer.py
Write a function def l2_reg_create_layer
(prev, n, activation, lambtha): that creates a tensorflow layer that includes L2 regularization:

prev is a tensor containing the output of the previous layer
n is the number of nodes the new layer should contain
activation is the activation function
that should be used on the layer
lambtha is the L2 regularization parameter
Returns: the output of the new layer
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates a tensorflow layer that includes l2 regularization"""
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n,
                            activation=activation,
                            kernel_initializer=initialize,
                            kernel_regularizer=regularizer)
    return layer(prev)
