#!/usr/bin/env python3
""" 0x02. Keras """
import tensorflow.keras as k


def test_model(network, data, labels, verbose=True):
    """Evaluates the output of a neural network."""
    # Evaluate the model on the test data
    results = network.evaluate(data, labels, verbose=verbose)

    # Extract the loss and accuracy from the evaluation results
    loss, accuracy = results[0], results[1]

    return loss, accuracy
