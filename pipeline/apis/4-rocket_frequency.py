#!/usr/bin/env python3
"""module 4-rocket_frequency
Returns the number of launches per rocket
"""
import requests
from collections import Counter


def fetch_launches_per_rocket():
    """Return the number of launches per rocket"""
    # Initialize rocket count dictionary
    rocket_count = Counter()

    # Fetch launches from SpaceX API
    response = requests.get("https://api.spacexdata.com/v3/launches")
    launches = response.json()

    # Fetch rocket details
    rocket_response = requests.get("https://api.spacexdata.com/v3/rockets")
    rockets = rocket_response.json()
    rocket_dict = {rocket['rocket_id']: rocket['rocket_name'] for rocket in rockets}

    # Count launches per rocket
    for launch in launches:
        rocket_id = launch['rocket']['rocket_id']
        rocket_name = rocket_dict.get(rocket_id, "Unknown")

        # Increment rocket count
        rocket_count[rocket_name] += 1

    # Sort by number of launches and then by rocket name
    sorted_rockets = sorted(rocket_count.items(), key=lambda x: (-x[1], x[0]))

    # Print the sorted rocket count
    for rocket, count in sorted_rockets:
        print(f"{rocket}: {count}")

if __name__ == "__main__":
    fetch_launches_per_rocket()
