#!/usr/bin/env python3
"""module 3-first_launch
write a script that displays
the first launch information.
"""
import requests
import sys
from datetime import datetime, timedelta

L_URL = "https://api.spacexdata.com/v4/launchpads/"
R_URL = "https://api.spacexdata.com/v4/rockets/"
LA_URL = "https://api.spacexdata.com/v4/launches"


def fetch_first_launch():
    """Return the first launch"""
    # Fetch launches from SpaceX API
    response = requests.get(LA_URL)
    launches = response.json()

    # Sort launches by date_unix
    sorted_launches = sorted(launches, key=lambda x: x['date_unix'])

    # Get the first launch
    first_launch = sorted_launches[0]

    # Extract required information
    launch_name = first_launch['name']
    launch_date = first_launch['date_local']
    rocket_id = first_launch['rocket']
    launchpad_id = first_launch['launchpad']

    # Fetch rocket details
    rocket_response = requests.get(R_URL + rocket_id)
    rocket_name = rocket_response.json()['name']

    # Fetch launchpad details
    launchpad_response = requests.get(L_URL + launchpad_id)
    launchpad_data = launchpad_response.json()
    launchpad_name = launchpad_data['name']
    launchpad_locality = launchpad_data['locality']

    # Print the information
    # Using join()
    print(" ".join([
        launch_name, "(", launch_date, rocket_name, ")", "-",
        launchpad_name, "(", launchpad_locality, ")"
    ]))

# Uncomment the following line to execute the function when running the script
# fetch_first_launch()
