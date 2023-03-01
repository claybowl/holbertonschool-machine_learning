#!/usr/bin/env python3
"""module 1-sensitivity.py
Write the function def sensitivity(confusion): that calculates the
sensitivity for each class in a confusion matrix:

confusion is a confusion numpy.ndarray of shape (classes, classes) where
row indices represent the correct labels and column indices represent the
predicted labels
classes is the number of classes
Returns: a numpy.ndarray of shape (classes,) containing
the sensitivity of each class
"""
import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each class in a confusion matrix"""
    classes = confusion.shape[0]
    true_positive = np.diag(confusion)
    false_negative = np.sum(confusion, axis=1) - true_positive
    sensitivity = true_positive / (true_positive + false_negative)
    return sensitivity
