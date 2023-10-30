#!/usr/bin/env python3
"""module 0-passengers
Fetches a list of ships from the API
"""
import requests


def availableShips(passengerCount):
    """Returns a list of ships available for a given
    passenger count"""
    # empty list to hold names of available ships
    available_ships = []

    # initialize URL for Swapi API endpoint for starships
    url = "https://swapi.dev/api/starships/"

    while url:
        # connect with API
        response = requests.get(url)
        ship_data = response.json()

        # loop through starship pages
        for starship in ship_data['results']:
            try:
                # Remove commas before converting to integer
                capacity_str = starship['passengers'].replace(",", "")
                capacity = int(capacity_str)
            except ValueError:
                continue  # Skip to the next iteration if conversion fails
            if capacity >= passengerCount:
                available_ships.append(starship['name'])

        # Move on to the next page of results, if any
        url = ship_data['next']

    return available_ships
