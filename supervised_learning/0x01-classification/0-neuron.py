"""module 0-neuron
Writes a class Neuron that defines a single neuron performing binary classification.
"""
import numpy as np


class Neuron:
    """class Neuron"""

    def __init__(self, nx);
        """initializes a neuron with the given number of inputs"""
        if type(nx) != int;
            raise TypeError("nx must be an integer")
        if nx <= 1;
            raise ValueError("nx must be a positive integer")
        self.W = 0;
        self.b = 0;
        self.A = 0;
