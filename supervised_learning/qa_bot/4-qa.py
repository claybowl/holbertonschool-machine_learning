#!/usr/bin/env python3
"""module 1-loop.py
"""
# Importing necessary libraries
import tensorflow_text as text
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Loading the Universal Sentence Encoder
use_model_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
use_model = hub.load(use_model_url)


def semantic_search(corpus_path, sentence):
    # Read the corpus
    corpus = []
    with open(corpus_path, 'r') as file:
        corpus = file.readlines()

    # Encode the sentence and documents
    sentence_embedding = use_model([sentence])
    document_embeddings = use_model(corpus)

    # Compute cosine similarity
    similarities = cosine_similarity(sentence_embedding, document_embeddings)

    # Find the most similar document
    most_similar_idx = np.argmax(similarities)
    most_similar_document = corpus[most_similar_idx]

    return most_similar_document


def question_answer(corpus_path):
    while True:
        question = input("Q: ").strip().lower()
        if question in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        else:
            # Perform semantic search to find the most relevant reference text
            reference = semantic_search(corpus_path, question)

            # Find the specific answer within the reference text
            answer = question_answer(question, reference)

            if answer:
                print("A:", answer)
            else:
                print("A: Sorry, I do not understand your question.")