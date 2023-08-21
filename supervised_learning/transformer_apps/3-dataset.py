#!/usr/bin/env python3
"""module 3-dataset.py
Create the class Dataset that
loads and preps a
dataset for machine translation
"""
import tensorflow_datasets as tfds
import tensorflow as tf


class Dataset:
    """class Dataset"""

    def __init__(self):
        """Construct"""
        examples, _ = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
        self.data_train = examples['train']
        self.data_valid = examples['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        # Training data pipeline
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(lambda pt, en: tf.logical_and(tf.size(pt) <= max_len,
                                                                              tf.size(en) <= max_len))
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(10000)
        self.data_train = self.data_train.padded_batch(batch_size, padded_shapes=([None], [None]))
        self.data_train = self.data_train.prefetch(tf.data.experimental.AUTOTUNE)

        # Validation data pipeline
        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(lambda pt, en: tf.logical_and(tf.size(pt) <= max_len,
                                                                              tf.size(en) <= max_len))
        self.data_valid = self.data_valid.padded_batch(batch_size, padded_shapes=([None], [None]))

    def tokenize_dataset(self, data):
        """tokenize_dataset"""
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """encodes a translation into tokens"""
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size+1]
        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size+1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """acts as a tensorflow wrapper for the encode instance method"""
        pt_tokens, en_tokens = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens

    def create_masks(inputs, target):
        """Creates all masks for training/validation"""
        # Encoder padding mask
        encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
        encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

        # Used in the 2nd attention block in the decoder
        decoder_mask = encoder_mask

        # Decoder padding mask
        dec_target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
        dec_target_padding_mask = dec_target_padding_mask[:, tf.newaxis, tf.newaxis, :]

        # Look ahead mask (to mask future tokens in the input received by the decoder)
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((tf.shape(target)[1], tf.shape(target)[1])), -1, 0)
        look_ahead_mask = look_ahead_mask[tf.newaxis, tf.newaxis, :, :]

        # Combined mask takes the maximum between look ahead mask and decoder target padding mask
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return encoder_mask, combined_mask, decoder_mask
