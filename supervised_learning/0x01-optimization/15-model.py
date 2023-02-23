#!/usr/bin/env python3
"""module 15-model
function that that builds, trains, and saves a neural network
model in tensorflow using Adam optimization, mini-batch gradient
descent, learning rate decay, and batch normalization"""
import tensorflow as tf
import numpy as np


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves a neural network model in
    Tensorflow using Adam optimization, mini-batch grad. descent,
    learning rate decay, and batch normalization.
    'Data_train' - tuple containing the training inputs/labels
    'Data_valid' - tuple containing the validation inputs/labels
    'layers' - list containing the number of nodes in each layer
    'activations' - list containing the activation functions
    'alpha' - learning rate
    'beta1' - weight for the first moment of Adam Optimization
    'beta2' - weight for the second moment of Adam Optimization
    epsilon - small number to avoid division by zero
    'decay_rate' - decay rate for inverse time decay of learning rate
    batch_size - number of data points that should be in a mini-batch
    epochs - number of times the training should pass through the whole dataset
    save_path - path where the model should be saved to
    """
    # Unpack the data
    (X_train, y_train) = Data_train
    (X_valid, y_valid) = Data_valid

    # Get the number of features and classes
    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]

    # Create placeholders for the input data and labels
    X = tf.placeholder(tf.float32, shape=[None, n_features], name="X")
    y = tf.placeholder(tf.float32, shape=[None, n_classes], name="y")

    # Create list to store weights and biases for each layer
    weights = []
    biases = []

    # Create a list to store the activations for each layer of the network
    activations = []

    # Create a placeholder for learning rate decay
    learningRateDecaySteps = tf.placeholder(tf.int32)

    # Build the network using layers and activations
    prevLayerInputs = X  # Set previous layer inputs to be input data

    for i in range(len(layers)):  # Iterate through each layer in layers list

        if i == 0:  # If first layer

            # Initialize weights with random normal distribution
            weights.append(tf.Variable(
                tf.random_normal([n_features, layers[i]])))

        else:

            # Initialize weights with random normal distribution
            weights.append(tf.Variable(
                tf.random_normal([layers[i-1], layers[i]])))

        # Initialize biases as zeros
        biases.append(tf.Variable(tf.zeros([layers[i]])))
        # Compute linear output of current layer
        linear_output = tf.add(
            tf.matmul(prevLayerInputs, weights[i]), biases[i])
        # Compute activation of current layer
        activations.append(activation(linear_output))
        # Set previous layer inputs to be the activations of the current layer
        prevLayerInputs = activations[i]

    # Define the loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=activations[-1]))

    # Define the optimizer
    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha, beta1=beta1,
        beta2=beta2, epsilon=epsilon).minimize(loss)

    # Define the initializer
    init = tf.global_variables_initializer()

    # Start the session
    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(epochs):

            epoch_loss = 0

            for batch in range(0, len(X_train), batch_size):

                batch_x = X_train[batch:batch+batch_size]
                batch_y = y_train[batch:batch+batch_size]

                _, batch_loss = sess.run([optimizer, loss], feed_dict={
                                         X: batch_x, y: batch_y,
                                         learningRateDecaySteps: decay_rate})

                epoch_loss += batch_loss

            # Evaluate the model after each epoch
            train_accuracy = np.mean(np.argmax(y_train, axis=1) == sess.run(
                tf.argmax(activations[-1], axis=1), feed_dict={X: X_train, y: y_train}))
    validation_accuracy = np.mean(np.argmax(y_valid, axis=1) == sess.run(
        tf.argmax(activations[-1], axis=1), feed_dict={X: X_valid, y: y_valid}))
    print("Epoch:", epoch+1, "Loss:", epoch_loss, "Train Accuracy:",
          train_accuracy, "Validation Accuracy:", validation_accuracy)

    # Save the model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_path)

    return save_path
