#!/usr/bin/env python3
"""Module policy_gradient
Write a function that computes to
policy with a weight of a matrix.
"""
import numpy as np


def policy(matrix, weight):
    """Computes the policy with weight of a matrix"""
    z = matrix.dot(weight)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def policy_gradient(state, weight):
    """
    Compute the Monte Carlo policy gradient
    based on state/weight.
    """
    probs = policy(state, weight)
    action =  np.random.choice(2, p=probs[0])
    dlog = np.zeros_like(probs)
    dlog[0, action] = 1
    grad = dlog - probs
    return action, grad
