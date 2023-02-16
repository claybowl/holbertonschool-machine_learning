#!/usr/bin/env python3
"""module 7-evaluate.py
Evaluates the output of a neural network.
"""
import tensorflow as tf
import numpy as np


def evaluate(X, Y, save_path):
    """evaluates the output of a neural network"""
    # Import modules
    create_placeholders = __import__(
        '0-create_placeholders').create_placeholders
    forward_prop = __import__('2-forward_prop').forward_prop
    create_train_op = __import__('5-create_train_op').create_train_op
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
    layer_sizes, activations = create_placeholders(X.shape[1], Y.shape[1])

    # Evaluation of output
    x, y = create_placeholders(X.shape[1], Y.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)

    # Engage Tensorflow session
    with tf.Session() as sess:
        # Restore saved model
        saver = tf.train.Saver()
        saver.restore(sess, save_path)

        # Evaluate model
        feed_dict = {x: X, y: Y}
        y_pred_hot, accuracy_val, loss_val = sess.run(
            [y_pred, accuracy, loss], feed_dict=feed_dict)

    return y_pred_hot, accuracy_val, loss_val
