#!/usr/bin/env python3
"""module 30-all
Function that lists all documents
in a collection
"""


def list_all(mongo_collection):
    """list all documents in a collection"""
    return mongo_collection.find()
