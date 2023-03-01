#!/usr/bin/env python3
"""module 2-l2_reg_cost.py
Write the function def l2_reg_cost(cost): that
calculates the cost of a neural network with L2 regularization:

cost is a tensor containing the cost of the
network without L2 regularization
Returns: a tensor containing the cost of the
network accounting for L2 regularization
"""
import tensorflow as tf


def l2_reg_cost(cost, weights, lambtha, L):
    """calculates the cost of a neural network with L2 regularization"""
    return cost + tf.losses.get_regularization_losses()
