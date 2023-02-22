#!/usr/bin/env python3
"""module 9-Adam
Updates a variable in place
using the Adam optimization algorithm.
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
	"""updates a variable in place using the Adam optimization algorithm"""
	v = beta1 * v + (1 - beta1) * grad
	s = beta2 * s + (1 - beta2) * (grad ** 2)
	vdw_corrected = v / (1 - beta1 ** t)
	sdw_corrected = s / (1 - beta2 ** t)
	var = var - alpha * vdw_corrected / (sdw_corrected ** 0.5 + epsilon)
	return var, v, s
