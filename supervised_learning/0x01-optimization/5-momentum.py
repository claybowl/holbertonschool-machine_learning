#!/usr/bin/env python3
"""module 5-momentum
Updates a variable using gradient descent with
momentum optimization algorithm.
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using gradient descent with
    momentum optimization algorithm
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
