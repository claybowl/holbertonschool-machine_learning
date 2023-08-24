#!/usr/bin/env python3
"""module 0-dataset.py
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
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
#       self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """tokenize_dataset"""
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en


if __name__ == "__main__":
    import tensorflow as tf

    data = Dataset()
    for pt, en in data.data_train.take(1):
        print(pt.numpy().decode('utf-8'))
        print(en.numpy().decode('utf-8'))
    for pt, en in data.data_valid.take(1):
        print(pt.numpy().decode('utf-8'))
        print(en.numpy().decode('utf-8'))
    print(type(data.tokenizer_pt))
    print(type(data.tokenizer_en))