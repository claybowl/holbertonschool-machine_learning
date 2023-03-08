#!/usr/bin/env python3
""" 0x02. Keras """
import tensorflow.keras as k


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix."""
    if classes is None:
        classes = len(np.unique(labels))
    one_hot_matrix = np.zeros((labels.shape[0], classes))
    one_hot_matrix[np.arange(labels.shape[0]), labels] = 1
    return one_hot_matrix
