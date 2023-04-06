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

def create_model():
    """creates the model for training"""
    base_model = ResNet50(weights='imagenet', input_shape=(224,224, 3), include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    resize_layer = Lambda(lambda image: tf.image.resize(image, (224,224)))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(10, activation='softmax')(x)

    # Create new model with our custum top layers
    model = Model(inputs=base_model.input, outputs=output)

    # Compile model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

    # load and pre-process CIFAR-10 dataset
if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # create model
    model = create_model()

    # train model
    datagen = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(X_train)
    history = model.fit(datagen.flow(X_train, Y_train, batch_size=64),
                        epochs=50, validation_data=(X_test, Y_test))

    # save model
    model.save('cifar10.h5')

    # evaluate model
    _, val_acc = model.evaluate(X_test, Y_test, verbose=0)
    print("Validation accuracy: {:.2f}%".format(val_acc * 100))

    # Check if the validation accuracy is 87% or higher
    assert val_acc >= 0.87, "The validation accuracy should be 87% or higher."
