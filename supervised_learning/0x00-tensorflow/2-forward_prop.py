#!/usr/bin/env python3
"""module 2-forward_prop.py
creates the forward propagation graph for the neural network.
"""
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates forward propagation graph"""
    # Imports 'create_layer' function
    create_layer = __import__('1-create_layer').create_layer
    # Assigns placeholder for input data
    prev = x
    # Iterates through layers
    for i in range(len(layer_sizes)):
        # Creates layer based on inputs
        prev = create_layer(prev, layer_sizes[i], activations[i])
    # Passing the output to next layer
    return prev
