#!/usr/bin/env python3
"""module 6-dropout_create_layer.py
Write a function
def dropout_create_layer(prev, n, activation, keep_prob):
that creates a layer of a neural network using dropout:

prev is a tensor containing the output of the previous layer
n is the number of nodes the new layer should contain
activation is the activation function that should be used on the layer
keep_prob is the probability that a node will be kept
Returns: the output of the new layer
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    dense = tf.layers.dense(prev, n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=regularizer)
    dropout = tf.layers.dropout(dense, rate=(1-keep_prob))
    return dropout
