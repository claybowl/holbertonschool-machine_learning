#!/usr/bin/env python3
"""module 31-insert_school
Inserts new python document in a collection
based on kwargs.
"""


def insert_school(mongo_collection, kwargs):
    """inserts a new document into collection"""
    # Insert the document and get the inserted document's ID
    inserted_doc = mongo_collection.insert_one(kwargs)
    
    # Return the new _id
    return mongo_collection.insert_one(kwargs).inserted_id
