#!/usr/bin/env python3
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Generator class for Generative Adversarial Networks (GANs).

        input_size: integer, size of the input vector
        hidden_size: integer, size of the hidden layer
        output_size: integer, size of the output vector
        """
        super(Generator, self).__init__()

        # Define the feed-forward neural network
        self.network = nn.Sequential(
            # First layer
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),

            # Second layer
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),

            # Output layer
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """
        Defines the forward pass of the generator.

        x: input tensor
        """
        return self.network(x)


# Example usage:
input_size = 100
hidden_size = 128
output_size = 784  # for generating 28x28 images
generator = Generator(input_size, hidden_size, output_size)

# Generate data
random_data = torch.randn((5, input_size))
generated_data = generator(random_data)
print(generated_data)
