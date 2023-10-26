#!/usr/bin/env python3
"""module 1-sentience
Returns a list of planets with sentient life
"""
import requests


def sentientPlanets():
    """Returns a list of planets with sentient life"""
    # initialize an empty list of sentient planets
    sentient_planets = []

    # create API object
    url = "https://swapi-api.hbtn.io/api/species/"

    while url:
        # make get request to url and store response
        # parse json data from response and store in data variable
        response = requests.get(url)
        sentience_data = response.json()

        # the loop for life
        for species in sentience_data['results']:
            if species['designation'] == 'sentient' or species['classification'] == 'sentient':
                homeworld_url = species['homeworld']

                # get the name of the planet from the homeworld url
                if homeworld_url:
                    homeworld_response = requests.get(homeworld_url)
                    homeworld_response = requests.get(homeworld_url).json()
                    sentient_planets.append(homeworld_response['name'])
        url = sentience_data['next']
    return sentience_data
