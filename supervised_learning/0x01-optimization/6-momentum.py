#!/usr/bin/env python3
"""module 6-momentum
Creates the training operation for a neural network
in tensorflow using the gradient descent with momentum
optimization algorithm.
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """creates the training operation using g.d. with momentum"""
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    train_op = optimizer.minimize(loss)
    return train_op
