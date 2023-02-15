#!/usr/bin/env python3
"""module 4-calculate_loss.py
Calculates the softmax cross-entropy loss of a prediction.
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss"""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y)
    return tf.reduce_mean(loss)
