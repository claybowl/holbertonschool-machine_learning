#!/usr/bin/env python3
"""module 0-vanilla
contains the function autoencoder
"""
from keras.layers import Input, Dense
from keras.models import Model



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
    encoder_input = Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = Dense(nodes, activation='relu')(x)
    latent = Dense(latent_dims, activation='relu')(x)
    encoder = Model(encoder_input, latent)

    # Decoder
    decoder_input = Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = Dense(nodes, activation='relu')(x)
    output = Dense(input_dims, activation='sigmoid')(x)
    decoder = Model(decoder_input, output)

    # Autoencoder
    autoencoder_input = Input(shape=(input_dims,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    auto = Model(autoencoder_input, decoded)

    # Compile autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
