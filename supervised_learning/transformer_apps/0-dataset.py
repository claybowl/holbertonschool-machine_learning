#!/usr/bin/env python3
"""module 0-dataset.py
Create the class Dataset that
loads and preps a
dataset for machine translation
"""
import tensorflow as tf


class Dataset:
    """class Dataset"""

    def __init__(self):
        self.data_train = tf.data.Dataset.from_tensor_slices(tf.config.experimental.get_memory_info(tf.data.Dataset.train_split_slices('ted_hrlr_translate/pt_to_en')))
        self.data_valid = tf.data.Dataset.from_tensor_slices(tf.config.experimental.get_memory_info(tf.data.Dataset.validate_split_slices('ted_hrlr_translate/pt_to_en')))
        self.tokenizer_pt = tf.keras.preprocessing.text.Tokenizer(num_words=2**15)
        self.tokenizer_en = tf.keras.preprocessing.text.Tokenizer(num_words=2**15)

    def tokenize_dataset(self, data):
        pt = data.pt
        en = data.en

        tokenizer_pt.add_tokens(pt.numpy())
        tokenizer_en.add_tokens(en.numpy())

        return self.tokenizer_pt, self.tokenizer_en
