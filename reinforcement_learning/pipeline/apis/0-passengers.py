#!/usr/bin/env python3
"""module 0-passengers
Fetches a list of ships from the API
"""
import requests

def availableShips(passengerCount, page=1):
    url = f"https://swapi.dev/api/ships/?page={page}"

    response = requests.get(url)
    ships = response.json()['results']

    available_ships = []

    for ship in ships:
        if ship['passengers'] >= passengerCount:
            available_ships.append(ship)

    return available_ships
