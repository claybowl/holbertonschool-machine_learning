#!/usr/bin/env python3
"""module 0-vanilla
contains the function autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder
    
    input_dims: integer, dimensions of the model input
    hidden_layers: list, number of nodes for each hidden layer in the encoder
    latent_dims: integer, dimensions of the latent space representation
    
    Returns: encoder, decoder, auto
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """

    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(encoder_input, latent)

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, output)

    # Autoencoder
    autoencoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    auto = keras.Model(autoencoder_input, decoded)

    # Compile autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
