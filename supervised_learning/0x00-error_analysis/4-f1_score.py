#!/usr/bin/env python3
"""module 4-f1_score.py
Write the function def f1_score(confusion): that calculates
the F1 score of a confusion matrix:

confusion is a confusion numpy.ndarray of shape (classes, classes)
where row indices represent the correct labels and column
indices represent the predicted labels
classes is the number of classes
Returns: a numpy.ndarray of shape (classes,) containing the F1 score of each class
You must use sensitivity = __import__('1-sensitivity').sensitivity
and precision = __import__('2-precision').precision create previously
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the F1 score of a confusion matrix"""
    # import sensitivity and precision functions
    sensitivity = sensitivity(confusion)
    precision = precision(confusion) 

    # Calculate F1 score
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    return f1
