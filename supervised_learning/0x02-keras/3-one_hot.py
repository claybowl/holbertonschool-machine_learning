#!/usr/bin/env python3
""" 0x02. Keras """
import tensorflow.keras as k


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Arguments:
    - labels: a numpy.ndarray with shape (m,) where m is the number of data points.
    - classes: an integer that represents the number of classes. If None, it will use the number of unique values in labels.

    Returns:
    a numpy.ndarray with shape (m, classes) representing the one-hot matrix.
    """
    if classes is None:
        classes = len(np.unique(labels))
    one_hot_matrix = np.zeros((labels.shape[0], classes))
    one_hot_matrix[np.arange(labels.shape[0]), labels] = 1
    return one_hot_matrix
