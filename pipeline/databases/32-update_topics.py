#!/usr/bin/env python3
"""module 31-update_topics
Updates topics based on name
"""

def update_topics(mongo_collection, name, topics):
    """
    Changes all topics of a school document based on the name
    """
    # Use the update_one method to update the topics of the school with the given name
    mongo_collection.update_one({"name": name}, {"$set": {"topics": topics}})
