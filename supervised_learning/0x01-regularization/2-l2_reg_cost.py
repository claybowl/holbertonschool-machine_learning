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
    l2_reg = 0
    for l in range(1, L + 1):
        l2_reg += tf.reduce_sum(
            tf.square(weights['W' + str(l)]))

    l2_reg = lambtha / (2 * m) * l2_reg
    cost = cost + l2_reg

    return cost
