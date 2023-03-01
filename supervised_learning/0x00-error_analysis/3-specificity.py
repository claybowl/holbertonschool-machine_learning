#!/usr/bin/env python3
"""module 3-specificity.py
Write the function def specificity(confusion): that calculates the specificity for each class in a confusion matrix:

confusion is a confusion numpy.ndarray of shape (classes, classes) where row indices represent the correct labels and column indices represent the predicted labels
classes is the number of classes
Returns: a numpy.ndarray of shape (classes,) containing the specificity of each class
"""
import numpy as np


def specificity(confusion):
    """calculates the specificity for each class in a confusion matrix"""
    