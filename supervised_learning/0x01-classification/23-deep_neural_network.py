#!/usr/bin/env python3
"""module 23-deep_neural_network
Class Deep Neural Network
"""
import numpy as np
import matplotlib.pyplot as plt



class DeepNeuralNetwork:
    """Class Deep Neural Network"""

    def __init__(self, nx, layers):
        """initializes the neural network"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(1, self.L + 1):
            he_init = np.sqrt(2 / nx)
            self.weights['W' +
                         str(l)] = np.random.randn(layers[l - 1], nx) * he_init
            self.weights['b' + str(l)] = np.zeros((layers[l - 1], 1))
            nx = layers[l - 1]

    def forward_prop(self, X):
        """forward propagation"""
        self.__cache["A0"] = X
        for l in range(1, self.__L + 1):
            W = self.__weights["W" + str(l)]
            b = self.__weights["b" + str(l)]
            A_prev = self.__cache["A" + str(l - 1)]
            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache["A" + str(l)] = A

        return A, self.__cache

    @property
    def L(self):
        """getter for the number of layers"""
        return self.__L

    @property
    def cache(self):
        """getter for the cache"""
        return self.__cache

    @property
    def weights(self):
        """getter for the weights"""
        return self.__weights

    def cost(self, Y, A):
        """calculates the cost"""
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """evaluates the model"""
        cache = self.forward_prop(X)
        A = cache["A" + str(self.__L)]
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """gradient descent"""
        m = Y.shape[1]
        dZ = cache["A" + str(self.__L)] - Y
        for l in range(self.__L, 0, -1):
            A = cache["A" + str(l - 1)]
            W = self.__weights["W" + str(l)]
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dW = (1 / m) * np.matmul(dZ, A.T)
            self.__weights["W" + str(l)] -= alpha * dW
            self.__weights["b" + str(l)] -= alpha * db
            dZ = np.matmul(W.T, dZ) * (A * (1 - A))

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            self.gradient_descent(Y, cache, alpha)

            if i % step == 0 or i == iterations:
                costs.append(cost)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

        if graph:
            iterations = np.arange(0, iterations + 1, step)
            plt.plot(iterations, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
