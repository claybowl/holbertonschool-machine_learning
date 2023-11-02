#!/usr/bin/env python3
"""module 33-school_by_topic
Function that returns the list of school having a specific topic
"""


def schools_by_topic(mongo_collection, topic):
    """Returns the list of school having a specific topic"""
    return mongo_collection.find({"topics": topic})
