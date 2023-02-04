#!/usr/bin/env python3
"""module 5-neuron
Writes a class Neuron that defines a single neuron
performing binary classification.
"""
import numpy as np


class Neuron():
    """class Neuron"""

    def __init__(self, nx):
        """initializes a neuron with the given number of inputs"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Returns the current weights of the neuron

        Returns:
        numpy.ndarray: The current weights of the neuron
        """
        return self.__W

    @property
    def b(self):
        """
        Returns the current bias of the neuron

        Returns:
        float: The current bias of the neuron
        """
        return self.__b

    @property
    def A(self):
        """
        Returns the current activation of the neuron

        Returns:
        float: The current activation of the neuron
        """
        return self.__A

    def forward_prop(self, X):
        """calculates the forward propagation of the neuron"""
        # Calculates linear activation.
        Z = np.matmul(self.__W, X) + self.__b
        # sigmoid activation function
        sigmoid = (1 / (1 + np.exp(-Z)))
        # Weighted sum Z passed through activation function
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """calculates the cost model using logistic regression"""
        # m is number of examples
        m = Y.shape[1]
        # formula calculating cost
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) *
                                 np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """evaluates the models predictions

        Parameters
        ----------
        X : numpy.ndarray with shape (nx, m) that contains
        the input data.
        Y : numpy.ndarray with shape (1, m) that contains
        the correct labels for input data

        Returns
        -------
        numpy.ndarray with shape (1, m) that contains predicted
        labels for each example.
        Label values should be 1 if output of network is
        >= 0.5. Otherwise, 0.
        """
        # computes calculations
        A = self.forward_prop(X)
        # converts calculations to predictions
        predict = np.where(A >= 0.5, 1, 0)
        # calculates cost of network
        cost = self.cost(Y, A)
        return predict, cost

    def gradient_descent(self, X, Y, A, alpha-0.05):
        """Calculates one iteration of gradient descent
        algorithm to update weights '__W' and bias '__b'
        of neuron.

        Parameters
        ----------
        X : numpy.ndarray with shape (nx, m) that contains
        the input data.
        Y : numpy.ndarray with shape (1, m) that contains
        the correct labels for input data
        A : numpy.ndarray with shape (1, m) containing
        the activated output of neuron.
        alpha: learning rate.

        Returns
        -------
        Updates the private attributes __W and __b.
        """
        # m is number of examples
        m = Y.shape[1]
        # calculates gradient
        dW = (1 / m) * np.dot(X, (A - Y).T)
        db = (1 / m) * np.sum(A - Y)
        # updates weights and bias
        self.__W -= alpha * dW
        self.__b -= alpha * db
