#!/usr/bin/env python3
"""module 11-learning_rate_decay
Updates the learning rate using inverse
time decay in numpy
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay in numpy"""
    return alpha / (1 + decay_rate * (global_step // decay_step))
