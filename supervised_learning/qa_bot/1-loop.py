#!/usr/bin/env python3
"""module 1-loop.py
"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
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
