#!/usr/bin/env python3
"""module 5-train_GAN
contains the function train_gan
"""
import torch
import torch.nn as nn
import torch.optim as optim


class Generator(nn.Module):
    # ... (as defined previously)


class Discriminator(nn.Module):
    # ... (as defined previously)

	def train_gan():
		"""
		Trains a Generative Adversarial Network (GAN).

		Returns: fake generated distribution from the Generator
		"""

		# Parameters
		learning_rate = 1e-3
		batch_size = 512
		iterations = 5000
		steps = 20
		d_input_size = 784
		g_input_size = 100
		hidden_size = 128
		output_size = 1

		# Create Generator and Discriminator
		Gen = Generator(g_input_size, hidden_size, d_input_size)
		Dis = Discriminator(d_input_size, hidden_size, output_size)

		# Loss function
		crit = nn.BCELoss()

		# Optimizers
		d_optimizer = optim.SGD(Dis.parameters(), lr=learning_rate)
		g_optimizer = optim.SGD(Gen.parameters(), lr=learning_rate)

		# Training
		for iteration in range(iterations):

			# Train Discriminator
			for _ in range(steps):
				d_optimizer.zero_grad()
				real_data = torch.normal(0, 1, size=(batch_size, d_input_size))
				fake_data = Gen(torch.randn(batch_size, g_input_size)).detach()
				prediction_real = Dis(real_data)
				error_real = crit(prediction_real, torch.ones(batch_size, 1))
				prediction_fake = Dis(fake_data)
				error_fake = crit(prediction_fake, torch.zeros(batch_size, 1))
				total_error = error_real + error_fake
				total_error.backward()
				d_optimizer.step()

			# Train Generator
			for _ in range(steps):
				g_optimizer.zero_grad()
				fake_data = Gen(torch.randn(batch_size, g_input_size))
				prediction = Dis(fake_data)
				error = crit(prediction, torch.ones(batch_size, 1))
				error.backward()
				g_optimizer.step()

		# Generate fake data
		fake_data = Gen(torch.randn(batch_size, g_input_size))

		return fake_data


	# Example usage:
	fake_data = train_gan()
	print(fake_data)
