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

    # Create placeholders for x and y.

    # Forward propagation.
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    # Calculates loss and accuracy of network predictions.

    # Create training operation.

    # Initialize variables.

