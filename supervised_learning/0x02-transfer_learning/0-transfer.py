#!/usr/bin/env python3
"""
Transfer Learning
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Lambda, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def preprocess_data(X, Y):
    """preprocesses the data for your model"""
    X_p = X.astype('float32')
    X_p /= 255
    X_p = tf.image.resize(X_p, (224, 224)).numpy()

    Y_p = to_categorical(Y, 10)

    return X_p, Y_p
