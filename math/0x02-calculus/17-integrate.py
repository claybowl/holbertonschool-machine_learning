#!/usr/bin/env python3
"""module 17-integrate
calculates derivative of poly"""


def poly_derivative(poly):
    """ Calculates the derivative of poly"""
    if type(poly) is not list or len(poly) <= 0:
        return None
    if len(poly) == 1:
        return [0]

    for power in range(len(poly)):
        poly[power] = (poly[power] * power)
    poly = poly[1:]
    return
