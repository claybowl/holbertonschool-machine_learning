#!/usr/bin/env python3
"""module 1-loop.py
"""
exit_commands = ['exit', 'quit', 'goodbye', 'bye']
while(True):
    d = input('Q: ')
    if d.lower() in exit_commands:
        print('A: Goodbye')
        break
    print("A: ")
