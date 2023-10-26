#!/usr/bin/env python3
"""module 4-rocket_frequency
Returns the number of launches per rocket
"""
import requests
from collections import Counter


def fetch_launches_per_rocket():
    # Initialize rocket count dictionary
    rocket_count = Counter()

    # Fetch launches from SpaceX API
    response = requests.get("https://api.spacexdata.com/v3/launches")
    launches = response.json()

    # Count launches per rocket
    for launch in launches:
        rocket_id = launch['rocket']['rocket_id']

        # Fetch rocket details
        rocket_response = requests.get(f"https://api.spacexdata.com/v3/rockets/{rocket_id}")
        rocket_name = rocket_response.json()['rocket_name']

        # Increment rocket count
        rocket_count[rocket_name] += 1

    # Sort by number of launches and then by rocket name
    sorted_rockets = sorted(rocket_count.items(), key=lambda x: (-x[1], x[0]))

    # Print the sorted rocket count
    for rocket, count in sorted_rockets:
        print(f"{rocket}: {count}")
