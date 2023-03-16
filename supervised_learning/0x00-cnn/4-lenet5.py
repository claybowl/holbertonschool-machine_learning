#!/usr/bin/env python3
"""Function that builds a modified version of the
LeNet-5 architecture using tensorflow"""
import tensorflow as tf


def lenet5(x, y):
    he_initializer = tf.contrib.layers.variance_scaling_initializer()

    # Convolutional layer 1
    conv1 = tf.layers.conv2d(x, filters=6, kernel_size=(
        5, 5), padding='same', activation=tf.nn.relu, kernel_initializer=he_initializer)

    # Max pooling layer 1
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2))

    # Convolutional layer 2
    conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=(
        5, 5), padding='valid', activation=tf.nn.relu, kernel_initializer=he_initializer)

    # Max pooling layer 2
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2))

    # Flatten the output
    flat = tf.layers.flatten(pool2)

    # Fully connected layer 1
    fc1 = tf.layers.dense(
        flat, units=120, activation=tf.nn.relu, kernel_initializer=he_initializer)

    # Fully connected layer 2
    fc2 = tf.layers.dense(fc1, units=84, activation=tf.nn.relu,
                          kernel_initializer=he_initializer)

    # Output softmax layer
    logits = tf.layers.dense(fc2, units=10, kernel_initializer=he_initializer)
    output = tf.nn.softmax(logits)

    # Loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)

    # Optimizer
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return output, training_op, loss, accuracy
