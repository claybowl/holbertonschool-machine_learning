#!/usr/bin/env python3
"""module 4-calculate_loss.py
Calculates the softmax cross-entropy loss of a prediction.
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss"""
    loss = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return tf.reduce_mean(loss)
