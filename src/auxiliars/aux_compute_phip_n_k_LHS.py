# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:38:07 2024

@author: egarate
"""

import numpy as np
from src.auxiliars.aux_LHS_distances import *


def compute_phip_lhs(LHS_i, p, distance_method="L1"):
    
    n_points = LHS_i.shape[0]
    n_features = LHS_i.shape[1]
    
    d_i_s = np.zeros(n_points - 1)
    
    distance_matrix = np.zeros((n_points, n_points))
    
    for point in range(n_points - 1):
        distances = get_distance(LHS_i[point, :], LHS_i[(point + 1):n_points, :], distance_method)
        
        distance_matrix[point, (point + 1):n_points] = distances
        distance_matrix[(point + 1):n_points, point] = distances 
        
        d_i = np.sum(distances ** (-p))
        d_i_s[point] = d_i
    
    phip = np.sum(d_i_s) ** (1 / p)
    
    return phip