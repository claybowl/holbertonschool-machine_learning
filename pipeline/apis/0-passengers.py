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
                capacity = int(starship['passengers'])
            except ValueError:
                continue
            if capacity >= passengerCount:
                available_ships.append(starship['name'])
        url = ship_data['next']
    return available_ships
