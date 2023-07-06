#!/usr/bin/env python3
"""module 2-sample_Z
contains the function sample_Z
"""
import torch


def sample_Z(mu, sigma, sampleType):
    """
    Creates input for the generator and discriminator.

    mu: float, mean of the distribution
    sigma: float, standard deviation of the distribution
    sampleType: str, "G" for generator, "D" for discriminator

    Returns: torch.Tensor or 0
        torch.Tensor if parameters are correct, 0 otherwise
    """
    # Check if sampleType is valid
    if sampleType not in ["G", "D"]:
        return 0

    # Sample size
    sample_size = 100

    # Generate input for the discriminator
    if sampleType == "D":
        # Input data from a normal distribution
        return torch.normal(mean=mu, std=sigma, size=(sample_size,))

    # Generate input for the generator
    elif sampleType == "G":
        # Random sampling
        return torch.randn(sample_size)


# Example usage:
mu = 0.0
sigma = 1.0
sampleType = "D"  # Use "G" for generator, "D" for discriminator
samples = sample_Z(mu, sigma, sampleType)
print(samples)
