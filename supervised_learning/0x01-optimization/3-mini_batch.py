#!/usr/bin/env python3
"""Module 3-mini_batch
Trains a loaded neural network model using mini-batch
gradient descent.
"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains  a loaded neural network model using mini-batch"""
    # Load model from save path
    saver = tf.train.import_meta_graph(load_path + ".meta")
    with tf.Session() as sess:
        saver.restore(sess, load_path)

    # Get placeholders, tensors, and operations
    x = tf.get_collection("x")[0]
    y = tf.get_collection("y")[0]
    accuracy = tf.get_collection("accuracy")[0]
    loss = tf.get_collection("loss")[0]
    train_op = tf.get_collection("train_op")[0]

    # Obtaining the number of training and validation data points
    m_train = X_train.shape[0]
    m_valid = X_valid.shape[0]

    for epoch in range(epochs + 1):
        # Shuffle training data
        X_train, Y_train = shuffle_data(X_train, Y_train)

        # Calculate the cost and accuracy
        train_cost = train_accuracy = sess.run([loss, accuracy], feed_dict={
                              x: X_train, y: Y_train})
        valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={
                              x: X_valid, y: Y_valid})

        # Print cost and accuracy
        print("After {} epochs:".format(epoch))
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuracy))

        # Loop training data in mini-batches
        step = 0
        for i in range(0, m_train, batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            # Run one step of gradient descent on mini-batch
            sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

            # Calculate the cost and accuracy for current mini-batch
            step_cost, step_accuracy = sess.run([loss, accuracy], feed_dict={
                              x: X_batch, y: Y_batch})

            if step % 100 == 0:
                print("\tStep {}:".format(step))
                print("\t\tCost: {}".format(step_cost))
                print("\t\tAccuracy: {}".format(step_accuracy))
            step += 1

            # Save the model
            saver.save(sess, save_path)
            return save_path
