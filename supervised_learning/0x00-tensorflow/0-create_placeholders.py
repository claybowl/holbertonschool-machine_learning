#!/usr/bin/env python3
"""module 0-create_placeholders
Function that returns two placeholders, 
x and y, for a neural network.
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """returns two placeholders for neural network"""
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
