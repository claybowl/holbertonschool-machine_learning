#!/usr/bin/env python3
"""Module 10-Adam
Creates the training operation for a neural
network in tensorflow using the Adam optimization algorithm.
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """training operation using Adam optimization algorithm"""
    # Creation of variables
    t = tf.Variable(0, dtype=tf.int32)
    v = tf.Variable(tf.zeros_like(loss), dtype=tf.float32)
    s = tf.Variable(tf.zeros_like(loss), dtype=tf.float32)

    # Calculation of corrected variables for adjusting 'v' and 's'
    v_corrected = v / (1 - beta1**(tf.cast(t, tf.float32)))
    s_corrected = s / (1 - beta2**(tf.cast(t, tf.float32)))

    # Calculation of gradient of the loss
    grad = tf.gradients(loss, tf.trainable_variables())

    # Updating 'v' and 's' and incrementing 't' by 1
    update_v = v.assign(beta1 * v + (1 - beta1) * grad)
    update_s = s.assign(beta2 * s + (1 - beta2) * grad**2)
    t = t.assign_add(1)

    # Calculation of Adam's optimization
    alpha_t = alpha * tf.sqrt(1 - beta2**t) / (1 - beta1**t)
    with tf.control_dependencies([update_v, update_s]):
        Adam_op = tf.train.AdamOptimizer(alpha_t).apply_gradients(
            zip(grad, tf.trainable_variables()))
    return Adam_op
