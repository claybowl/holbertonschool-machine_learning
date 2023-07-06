#!/usr/bin/env python3
"""module 4-train_generator
contains the function train_gen
"""
import torch
import torch.nn as nn


def train_gen(Gen, Dis, gInputSize, mbatchSize, steps, optimizer, crit):
    """
    Trains the generator of a GAN.

    Gen: Generator object
    Dis: Discriminator object
    gInputSize: int, input size of Generator input data
    mbatchSize: int, batch size for training
    steps: int, number of steps for training
    optimizer: torch.optim object, stochastic gradient descent optimizer
    crit: torch.nn object, BCELoss function

    Returns: error of the fake data, and the fake data set
    """

    for step in range(steps):

        # Zero the gradients
        optimizer.zero_grad()

        # Generate fake data with the generator
        noise = torch.randn(mbatchSize, gInputSize)
        fake_data = Gen(noise)

        # Calculate the error on fake data
        prediction = Dis(fake_data)
        error = crit(prediction, torch.ones(mbatchSize, 1))

        # Backward pass
        error.backward()

        # Update weights
        optimizer.step()

    return error, fake_data
#!/usr/bin/env python3
import torch
import torch.nn as nn

def train_gen(Gen, Dis, gInputSize, mbatchSize, steps, optimizer, crit):
    """
    Trains the generator of a GAN.

    Gen: Generator object
    Dis: Discriminator object
    gInputSize: int, input size of Generator input data
    mbatchSize: int, batch size for training
    steps: int, number of steps for training
    optimizer: torch.optim object, stochastic gradient descent optimizer
    crit: torch.nn object, BCELoss function

    Returns: error of the fake data, and the fake data set
    """

    for step in range(steps):

        # Zero the gradients
        optimizer.zero_grad()

        # Generate fake data with the generator
        noise = torch.randn(mbatchSize, gInputSize)
        fake_data = Gen(noise)

        # Calculate the error on fake data
        prediction = Dis(fake_data)
        error = crit(prediction, torch.ones(mbatchSize, 1))

        # Backward pass
        error.backward()

        # Update weights
        optimizer.step()

    return error, fake_data


# Example usage:
# Assuming Gen and Dis are instances of the Generator and Discriminator classes
# Assuming optimizer is an instance of torch.optim.SGD
# crit = nn.BCELoss()
# train_gen(Gen, Dis, gInputSize=100, mbatchSize=64, steps=1000, optimizer=optimizer, crit=crit)
#!/usr/bin/env python3


def train_gen(Gen, Dis, gInputSize, mbatchSize, steps, optimizer, crit):
    """
    Trains the generator of a GAN.

    Gen: Generator object
    Dis: Discriminator object
    gInputSize: int, input size of Generator input data
    mbatchSize: int, batch size for training
    steps: int, number of steps for training
    optimizer: torch.optim object, stochastic gradient descent optimizer
    crit: torch.nn object, BCELoss function

    Returns: error of the fake data, and the fake data set
    """

    for step in range(steps):

        # Zero the gradients
        optimizer.zero_grad()

        # Generate fake data with the generator
        noise = torch.randn(mbatchSize, gInputSize)
        fake_data = Gen(noise)

        # Calculate the error on fake data
        prediction = Dis(fake_data)
        error = crit(prediction, torch.ones(mbatchSize, 1))

        # Backward pass
        error.backward()

        # Update weights
        optimizer.step()

    return error, fake_data

# Example usage:
# Assuming Gen and Dis are instances of the Generator and Discriminator classes
# Assuming optimizer is an instance of torch.optim.SGD
# crit = nn.BCELoss()
# train_gen(Gen, Dis, gInputSize=100, mbatchSize=64, steps=1000, optimizer=optimizer, crit=crit)
