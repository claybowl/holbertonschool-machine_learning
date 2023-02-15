#!/usr/bin/env python3
"""module 6-train.py
Builds, trains, and saves a neural network classifier.
"""
import tensorflow as tf


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """Trains a neural network classifier.

    Arguments:
        X_train (numpy.ndarray): Training data.
        Y_train (numpy.ndarray): Training labels.
        X_valid (numpy.ndarray): Validation data.
        Y_valid (numpy.ndarray): Validation labels.
        layer_sizes: List of layer sizes.
        activations: List of activation functions.
        alpha: Learning rate.
        iterations: Number of iterations.
        save_path: Path to save the model.
    """
    # Import network functions.
    create_train_op = __import__('5-create_train_op').create_train_op
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
    create_placeholders = __import__('0-create_placeholders').create_placeholders
    forward_prop = __import__('2-forward_prop').forward_prop

    # Create placeholders for x and y.
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Forward propagation.
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculates loss and accuracy of network predictions.
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Create training operation.
    train_op = create_train_op(loss, alpha)

    # Initialize variables.
    init = tf.global_variables_initializer()

    # Tensorflow session
    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))

        saver = tf.train.Saver()
        saver.save(sess, save_path)
    return save_path
