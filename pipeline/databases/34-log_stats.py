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
    nginx_collection = client.logs.nginx

    # Count the total number of logs
    total_logs = nginx_collection.count_documents({})
    print("{} logs".format(total_logs))

    # Print the methods statistics
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        count = nginx_collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, count))

    # Count the number of documents with method=GET and path=/status
    status_check = nginx_collection.count_documents({"method": "GET", "path": "/status"})
    print("{} status check".format(status_check))
