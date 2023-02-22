#!/usr/bin/env python3
"""module 11-learning_rate_decay
A Python module that updates the learning rate
using inverse time decay in numpy.
`alpha` - the initial learning rate
`decay_rate` - the learning rate decay rate
`global_step` - the current step of training
`decay_step` - the step after which the learning rate should decay
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """takes the learning rate parameters and returns
    the current learning rate"""
    return alpha / (1 + decay_rate * (global_step // decay_step))
