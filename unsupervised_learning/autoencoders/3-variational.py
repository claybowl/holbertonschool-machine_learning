#!/usr/bin/env python3
"""module 2-convolutional
contains the function autoencoder
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    
    input_dims: integer, dimensions of the model input
    hidden_layers: list, number of nodes for each hidden layer in the encoder
    latent_dims: integer, dimensions of the latent space representation
    
    Returns: encoder, decoder, auto
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """

    # Encoder
    encoder_input = layers.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = layers.Dense(nodes, activation='relu')(x)
    z_mean = layers.Dense(latent_dims)(x)
    z_log_var = layers.Dense(latent_dims)(x)
    z = layers.Lambda(sampling, output_shape=(
        latent_dims,))([z_mean, z_log_var])
    encoder = Model(encoder_input, [z, z_mean, z_log_var])

    # Decoder
    decoder_input = layers.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = layers.Dense(nodes, activation='relu')(x)
    output = layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = Model(decoder_input, output)

    # Autoencoder
    autoencoder_input = layers.Input(shape=(input_dims,))
    z, _, _ = encoder(autoencoder_input)
    decoded = decoder(z)
    auto = Model(autoencoder_input, decoded)

    # Custom loss considering the KL divergence
    reconstruction_loss = tf.keras.losses.binary_crossentropy(
        autoencoder_input, decoded)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    # Compile autoencoder
    auto.compile(optimizer='adam')

    return encoder, decoder, auto


# Example usage:
input_dims = 784
hidden_layers = [256, 128]
latent_dims = 2
encoder, decoder, auto = autoencoder(input_dims, hidden_layers, latent_dims)
