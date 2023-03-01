#!/usr/bin/env python3
"""module 3-specificity.py
Write the function def specificity(confusion): that calculates the specificity for each class in a confusion matrix:

confusion is a confusion numpy.ndarray of shape (classes, classes) where row indices represent the correct labels and column indices represent the predicted labels
classes is the number of classes
Returns: a numpy.ndarray of shape (classes,) containing the specificity of each classu
"""
import numpy as np


def specificity(confusion):
    """calculates the specificity for each class in a confusion matrix"""
    classes = confusion.shape[0]
    specificity = np.zeros(classes)

    # loop through each class
    for i in range(classes):
        # Calculate True Negative (tn) and False Positive (fp)
        tn = np.delete(confusion, i, 0).sum() - \
            np.delete(confusion, i, 1)[:, i].sum()
        fp = confusion[:, i].sum() - confusion[i, i]
        # Calculate specificity for the current class
        specificity[i] = tn / (tn + fp)

    return specificity
