#!/usr/bin/env python3
"""module 7-evaluate.py
Evaluates the output of a neural network.
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """evaluates the output of a neural network"""
    # Create a tensorflow session
    sess = tf.Session()

    # Restore the saved model
    saver = tf.train.import_meta_graph(save_path + '.meta')
    saver.restore(sess, save_path)

    # Get the tensors from the saved model
    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]
    y_pred = tf.get_collection('y_pred')[0]
    accuracy = tf.get_collection('accuracy')[0]
    loss = tf.get_collection('loss')[0]

    # Evaluate the model
    feed_dict = {x: X, y: Y}
    y_pred_eval, accuracy_eval, loss_eval = sess.run(
        [y_pred, accuracy, loss], feed_dict=feed_dict)

    # Close the session
    sess.close()

    return y_pred_eval, accuracy_eval, loss_eval
