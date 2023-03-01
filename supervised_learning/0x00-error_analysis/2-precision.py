#!/usr/bin/env python3
"""module 2-precision.py
Write the function def precision(confusion): that calculates
the precision for each class in a confusion matrix:

confusion is a confusion numpy.ndarray of shape (classes, classes)
where row indices represent the correct labels and column indices
represent the predicted labels
classes is the number of classes
Returns: a numpy.ndarray of shape (classes,) containing the
precision of each class
"""
import numpy as np


def precision(confusion):
    """calculates the precision for each class in a confusion matrix"""
    classes = confusion.shape[0]
    precision = np.zeros((classes,))

    # Calculate precision for each class
    for i in range(classes):
        # Precision calculated as ratio of true positive to sum of true positive and false positive
        true_positive = confusion[i][i]
        false_positive = np.sum(confusion[:, i]) - true_positive
        precision[i] = true_positive / (true_positive + false_positive)

    return precision
