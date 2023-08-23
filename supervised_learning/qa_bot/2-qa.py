#!/usr/bin/env python3
"""module 0-qa.py
"""
# Importing necessary libraries
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import tensorflow_text as text
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Checking the installed versions
print('Tensorflow version:', tf.__version__)
print('Numpy version:', np.__version__)

# Loading the BERT model for question answering
bert_qa_model_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3'
bert_qa_model = hub.load(bert_qa_model_url)

# Loading the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


def preprocess_input(question, reference, tokenizer):
    """Preprocesses the question and reference text for the BERT model.

    Args:
        question (str): The question to answer.
        reference (str): The reference document from which to find the answer.
        tokenizer (BertTokenizer): The BERT tokenizer.

    Returns:
        dict: A dictionary containing the input IDs, attention masks, and token type IDs.
    """
    # Tokenize the question and reference
    inputs = tokenizer(question, reference, return_tensors='tf', max_length=512, truncation=True)

    # Create input IDs, attention masks, and token type IDs
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    # Prepare the model input
    model_input = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }

    return model_input


def question_answer(question, reference):
    """Finds a snippet of text within a reference document to answer a question.

    Args:
        question (str): The question to answer.
        reference (str): The reference document from which to find the answer.

    Returns:
        str: The answer as a string or None if no answer is found.
    """
    # Preprocess the input
    model_input = preprocess_input(question, reference, tokenizer)

    # Run the BERT model
    outputs = bert_qa_model(model_input)
    start_logits, end_logits = outputs['start_logits'], outputs['end_logits']

    # Find the answer span
    start_token_idx = tf.argmax(start_logits, axis=-1)[0]
    end_token_idx = tf.argmax(end_logits, axis=-1)[0]

    # Extract the answer text
    input_ids = model_input['input_ids'].numpy()[0]
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[start_token_idx:end_token_idx + 1])
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer if answer else None


def answer_loop(reference):
    """answers questions from a reference text"""
    while True:
        question = input("Q: ").strip().lower()
        if question in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        else:
            answer = question_answer(question, reference)
            if answer:
                print("A:", answer)
            else:
                print("A: Sorry, I do not understand your question.")
