#!/usr/bin/env python3
"""Module 10-Adam
Creates the training operation for a neural
network in tensorflow using the Adam optimization algorithm.
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """training operation using Adam optimization algorithm"""
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
