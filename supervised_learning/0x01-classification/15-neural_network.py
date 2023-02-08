#!/usr/bin/env python3
"""module 15-neural_network
Defines a neural network with one hidden layer performing binary classification
"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """class NeuralNetwork"""

    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def sigmoid(self, Z):
        """sigmoid function"""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """calculates the forward propagation"""
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(Z1)
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(Z2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        cost = -(np.matmul(Y, np.log(A).T) + np.matmul(1 -
                 Y, np.log(1.0000001 - A).T)) / Y.shape[1]
        cost = np.squeeze(cost)
        return cost

    def evaluate(self, X, Y):
        """evaluates the neural networks predictions"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """calculates one pass of gradient descent"""
        m = X.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(A1, dz2.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if (verbose or graph) and type(step) != int:
            raise TypeError("step must be an integer")
        if (verbose or graph) and (step <= 0 or step > iterations):
            raise ValueError("step must be positive and <= iterations")

        for i in range(iterations + 1):
            self.__A1, self.__A2 = self.forward_prop(X)
            if i % step == 0:
                if verbose:
                    print(
                        f"Cost after {i} iterations: {self.cost(Y, self.__A2)}")
                if graph:
                    if i == 0 or i == iterations:
                        plt.scatter(i, self.cost(Y, self.__A2), color='red')
                    else:
                        plt.scatter(i, self.cost(Y, self.__A2), color='blue')
            dA2 = -(np.divide(Y, self.__A2) - np.divide(1 - Y, 1 - self.__A2))
            dZ2 = self.__A2 - Y
            dW2 = np.matmul(dZ2, self.__A1.T) / X.shape[1]
            db2 = np.sum(dZ2, axis=1, keepdims=True) / X.shape[1]
            dZ1 = np.matmul(self.__W2.T, dZ2) * (1 - np.power(self.__A1, 2))
            dW1 = np.matmul(dZ1, X.T) / X.shape[1]
            db1 = np.sum(dZ1, axis=1, keepdims=True) / X.shape[1]
            self.__W2 -= alpha * dW2
            self.__b2 -= alpha * db2
            self.__W1 -= alpha * dW1
            self.__b1 -= alpha * db1

        if graph:
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
