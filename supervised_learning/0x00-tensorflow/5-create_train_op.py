#!/usr/bin/env python3
"""module 5-create_train_op.py
creates the training operation for the network
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """ creates the training operation for the network"""
    optimizer =  tf.train.GradientDescentOptimizer(learning_rate=alpha)
    return optimizer.minimize(loss)
