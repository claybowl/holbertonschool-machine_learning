#!/usr/bin/env python3
"""Module 1-ngram_bleu.py
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """Function that calculates the n-gram BLEU score for a sentence"""
    def make_ngrams(sentence, n):
        """converts a sentence to ngrams"""
        ngrams = []
        for i in range(0, len(sentence) - n + 1):
            grams = [sentence[i + j] for j in range(n)]
            ngram = ''
            for word in grams:
                ngram += word
                if word != grams[-1]:
                    ngram += ' '
            ngrams.append(ngram)

        return ngrams

    def count_ngrams(sentence, ngrams, n):
        ngram_sentence = make_ngrams(sentence, n)
        ngram_count = {ngram: 0 for ngram in ngrams}
        for ngram in ngrams:
            if ngram in ngram_sentence:
                ngram_count[ngram] += 1

        return ngram_count

    ngrams = make_ngrams(sentence, n)

    c = len(sentence)
    r = min([len(reference) for reference in references])
    BP = 1 if c > r else np.exp(1 - (r / c))

    max_ref_count = {ngram: 0 for ngram in ngrams}
    for reference in references:
        ngram_count = count_ngrams(reference, ngrams, n)
        for ngram in ngrams:
            max_ref_count[ngram] = max(ngram_count[ngram],
                                       max_ref_count[ngram])

    Pn = np.sum(list(max_ref_count.values())) / np.sum(
        list(count_ngrams(sentence, ngrams, n).values()))

    bleu = BP * Pn

    return bleu
