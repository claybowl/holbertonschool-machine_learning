#!/usr/bin/env python3
"""module 4-create_masks.py
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def create_masks(self, inputs, target):
    """creates all masks for training/validation"""
    # Encoder padding mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Decoder combined mask
    _max = self.max_len
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((_max, _max)), -1, 0)
    target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    decoder_mask = tf.maximum(look_ahead_mask, target_padding_mask)
    decoder_mask = decoder_mask[:, tf.newaxis, :, :]

    # Encoder padding mask (for the second attention block in the decoder)
    decoder_padding_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_padding_mask = decoder_padding_mask[:, tf.newaxis, tf.newaxis, :]

    return encoder_mask, decoder_mask, decoder_padding_mask


if __name__ == "__main__":
    Dataset = __import__('3-dataset').Dataset
    import tensorflow as tf

    tf.compat.v1.set_random_seed(0)
    data = Dataset(32, 40)
    for inputs, target in data.data_train.take(1):
        print(create_masks(data, inputs, target))
