# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:28:39 2024

@author: egarate
"""

import numpy as np

def get_n_permutation(n_points):
    permutation = np.random.permutation(n_points) + 1  
    return permutation


def create_random_n_k_lhs(n_points, n_features):
    LHS_i = np.zeros((n_points, n_features))
    
    for j_feature in range(n_features):
        LHS_i[:, j_feature] = get_n_permutation(n_points)
    
    return LHS_i
