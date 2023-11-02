#!/usr/bin/env python3
"""module 33-log_stats
Script that provides some stats about Nginx logs
stored in MongoDB
"""
from pymongo import MongoClient

if __name__ == "__main__":
    # Connect to the MongoDB client
    client = MongoClient('mongodb://127.0.0.1:27017')
    # Access the 'logs' database and 'nginx' collection
    db = client['logs']
    nginx_collection = db['nginx']

    # Count the total number of logs
    nginx_logs = nginx_collection.count_documents({})
    print(str(nginx_logs) + ' logs')
    print("Methods:")

    # Print the methods statistics
    method = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    method_stats_dict = {}

    for method in method:
        method_count = nginx_collection.count_documents({"method": method})
        method_stats_dict[method] = method_count

    for item, value in method_stats_dict.items():
        print('\tmethod ' + item + ': ' + str(value))

    status_count = nginx_collection.count_documents({'method': 'GET',
                                                     'path': "/status"})
    print(str(status_count) + " status check")
