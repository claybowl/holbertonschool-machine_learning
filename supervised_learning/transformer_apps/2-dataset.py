#!/usr/bin/env python3
"""module 2-dataset.py
Create the class Dataset that
loads and preps a
dataset for machine translation
"""
import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf


class Dataset:
    """class Dataset"""

    def __init__(self):
        """Construct"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

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


if __name__ == "__main__":
    import tensorflow as tf

    data = Dataset()
    print('got here')
    for pt, en in data.data_train.take(1):
        print(pt, en)
    for pt, en in data.data_valid.take(1):
        print(pt, en)
