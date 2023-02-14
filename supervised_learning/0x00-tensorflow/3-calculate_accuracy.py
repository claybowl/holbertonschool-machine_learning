#!/usr/bin/env python3
"""module 3-calculate_accuracy.py
Calculates the accuracy of a prediction
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction"""
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy
