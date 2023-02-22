#!/usr/bin/env python3
"""module 12-learning_rate_decay
Implements a learning rate decay for the given
hyperparameter `alpha'.
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """implements the tensorflow with the 'staircase' parameter
    set to true.
    `alpha`: The initial learning rate.
    `decay_rate`: The rate of decay.
    `global_step`: The step at which to start the decay.
    `decay_step`: The step size (number of global steps) between each decay.
    """
    updated_alpha = tf.train.inverse_time_decay(alpha, global_step, decay_step,
									   decay_rate, staircase=True)
    return updated_alpha
