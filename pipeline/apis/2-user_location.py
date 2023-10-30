#!/usr/bin/env python3
"""module 2-user_location git Returns the user location
"""
import requests
from datetime import datetime
import sys


def fetch_user_location(api_url):
    """fetches user location"""
    response = requests.get(api_url)

    # check if response is valid
    if response.status_code == 200:
        # Parse the JSON data from the response
        data = response.json()

        # Print the location of the user
        print(data.get('location', 'Location not set'))

    elif response.status_code == 404:
        # User does not exist
        print("Not found")

    elif response.status_code == 403:
        # Rate limit exceeded
        reset_time = int(response.headers.get('X-Ratelimit-Reset', 0))

        # Calculate the time remaining for the rate limit to reset
        reset_time = datetime.fromtimestamp(reset_time) - datetime.now()
        minutes = reset_time.total_seconds() // 60

        # Using string concatenation
        print("Reset in " + str(int(minutes)) + " min")


if __name__ == '__main__':
    fetch_user_location(sys.argv[1])
