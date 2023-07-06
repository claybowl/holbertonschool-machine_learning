#!/usr/bin/env python3
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Discriminator class for Generative Adversarial Networks (GANs).

        input_size: integer, size of the input vector
        hidden_size: integer, size of the hidden layer
        output_size: integer, size of the output vector
        """
        super(Discriminator, self).__init__()

        # Define the feed-forward neural network
        self.network = nn.Sequential(
            # First layer
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),

            # Second layer
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),

            # Output layer
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Defines the forward pass of the discriminator.

        x: input tensor
        """
        return self.network(x)


# Example usage:
input_size = 784  # for 28x28 images
hidden_size = 128
output_size = 1  # binary classification: real or fake
discriminator = Discriminator(input_size, hidden_size, output_size)

# Classify data
sample_data = torch.randn((5, input_size))
classification = discriminator(sample_data)
print(classification)
