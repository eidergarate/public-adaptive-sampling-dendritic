# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:26:36 2024

@author: egarate
"""

import numpy as np


def L1(xi, xj):
    distance = np.sum(np.abs(xi - xj))
    return distance

def euclidean(xi, xj):
    distance = np.sqrt(np.sum((xi - xj) ** 2))
    return distance


def get_distance(xi, others, distance_method="L1"):
    if distance_method == "L1":
        if isinstance(others, np.ndarray):
            distances = np.apply_along_axis(L1, 1, others, xi)
        else:
            distances = L1(xi, others)
    
    elif distance_method == "euclidean":
        if isinstance(others, np.ndarray):
            distances = np.apply_along_axis(euclidean, 1, others, xi)
        else:
            distances = euclidean(xi, others)
    
    return distances


def get_hamming_distance(x_vect, y_vect):
    return np.sum(x_vect != y_vect)
