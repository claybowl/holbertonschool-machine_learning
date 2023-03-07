#!/usr/bin/env python3
""" 0x02. Keras """
import tensorflow.keras as k


def optimize_model(network, alpha, beta1, beta2):
    """Sets up Adam optimization for a Keras model"""
    network.compile(optimizer=K.optimizers.Adam(lr=alpha, beta_1=beta1,
                    beta_2=beta2), loss='categorical_crossentropy', metrics=['accuracy'])
