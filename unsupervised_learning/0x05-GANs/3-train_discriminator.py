#!/usr/bin/env python3
"""module 3-train_discriminator
contains the function train_dis
"""
import torch
import torch.nn as nn


def train_dis(Gen, Dis, dInputSize, gInputSize, mbatchSize, steps, optimizer, crit):
    """
    Trains the discriminator of a GAN.

    Gen: Generator object
    Dis: Discriminator object
    dInputSize: int, input size of Discriminator input data
    gInputSize: int, input size of Generator input data
    mbatchSize: int, batch size for training
    steps: int, number of steps for training
    optimizer: torch.optim object, stochastic gradient descent optimizer
    crit: torch.nn object, BCELoss function

    Returns: error estimate of the fake and real data, along with the fake and real data sets
    """

    for step in range(steps):

        # Zero the gradients
        optimizer.zero_grad()

        # Generate real data with normal distribution
        mu, sigma = 0, 1
        real_data = torch.normal(mu, sigma, size=(mbatchSize, dInputSize))

        # Generate fake data with the generator
        noise = torch.randn(mbatchSize, gInputSize)
        fake_data = Gen(noise).detach()

        # Calculate the error on real data
        prediction_real = Dis(real_data)
        error_real = crit(prediction_real, torch.ones(mbatchSize, 1))

        # Calculate the error on fake data
        prediction_fake = Dis(fake_data)
        error_fake = crit(prediction_fake, torch.zeros(mbatchSize, 1))

        # Total error
        total_error = error_real + error_fake

        # Backward pass
        total_error.backward()

        # Update weights
        optimizer.step()

    return error_real, error_fake, real_data, fake_data

# Example usage:
# Assuming Gen and Dis are instances of the Generator and Discriminator classes
# Assuming optimizer is an instance of torch.optim.SGD
# crit = nn.BCELoss()
# train_dis(Gen, Dis, dInputSize=784, gInputSize=100, mbatchSize=64, steps=1000, optimizer=optimizer, crit=crit)
