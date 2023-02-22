#!/usr/bin/env python3
"""module 14-batch_norm
function creates a batch normalization layer for a
neural network in TensorFlow.
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """tensor that represents the activation of
    a normalized batch.
    `prev`: tensor containing the output of the previous layer
    `n`: int representing the number of nodes in the layer
    `activation`: function or string to be applied as activation function
    """
    # Initialize the weights of layers using variance scaling
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    # Creates a dense layer with 'n' neurons, no activation
    dense_layer = tf.layers.Dense(units=n, activation=None,
                                  kernel_initializer=initializer)
    Z = dense_layer(prev)

    # Calculate the mean and variance of layer output
    mean, var = tf.nn.moments(Z, axes=[0], keep_dims=True)

    # Initialize gamma and beta for normalization
    gamma = tf.Variable(tf.ones([1, n]))
    beta = tf.Variable(tf.zeros([1, n]))
    epsilon = 1e-8

    # Applies batch normalization to the output
    z_norm = tf.nn.batch_normalization(Z, mean, var, beta, gamma, epsilon)

    # Returns the activation of the normalized layer
    return activation(z_norm)
