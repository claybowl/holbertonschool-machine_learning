#!/usr/bin/env python3
"""module 8-RMSProp
Creates the training operation for a neural network in
tensorflow using the RMSProp optimization algorithm.
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """creates the training operation using the RMSProp
    optimization algorithm"""
    optimizer = tf.train.RMSPropOptimizer(
                  learning_rate=alpha, decay=beta2, epsilon=epsilon)
    return tf.train.RMSPropOptimizer(alpha, beta2, epsilon).minimize(loss)
