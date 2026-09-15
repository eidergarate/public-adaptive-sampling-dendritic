# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:22:15 2024

@author: egarate
"""

import numpy as np
import pandas as pd
from src.Optimal_LHS.LaPSO_LHS import LaPSO_LHS 
from datetime import datetime

def get_sampling(partition_matrix, OLHS):
    
    sampling = pd.DataFrame(np.zeros((OLHS['last_LHS'].shape[0], OLHS['last_LHS'].shape[1] - 1)))
    sampling.columns = partition_matrix.columns

    Optimal_LHS = OLHS['last_LHS'].copy()
    
    Optimal_LHS.reset_index(drop = True, inplace = True)

    for row in range(sampling.shape[0]):
        for column in range(sampling.shape[1]):
            
            feature_idx = 'feature' + str(column + 1)
            feature_name = partition_matrix.columns[column]
            sampling.iloc[row, column] = partition_matrix.loc[int(Optimal_LHS.loc[row, feature_idx] - 1), feature_name]

    return sampling

def find_OLHS_PSO(range_matrix, n_points, n_pop, max_iter, tol, distance_method, p, R):
    
    n_features = range_matrix.shape[1]
    partition = (range_matrix.iloc[1, :] - range_matrix.iloc[0, :]) / (n_points - 1)

    
    OLHS = LaPSO_LHS(n_points, n_features, n_pop, max_iter, tol, distance_method, p, R)

    partition_matrix = pd.DataFrame(np.zeros((n_points, n_features)))
    for j_var in range(n_features):
        partition_matrix.iloc[:, j_var] = np.linspace(
            range_matrix.iloc[0, j_var], range_matrix.iloc[1, j_var], n_points
        )

    partition_matrix.columns = range_matrix.columns

    LHS_sampling = get_sampling(partition_matrix, OLHS)

    return LHS_sampling

def get_sampling_j_aniso(n_points):
    rand_j_aniso = np.random.choice(np.arange(1, n_points + 1), size=n_points // 2, replace=False)
    sampling_j_aniso = np.zeros(n_points)

    sampling_j_aniso[rand_j_aniso - 1] = 4
    sampling_j_aniso[sampling_j_aniso == 0] = 6

    return sampling_j_aniso

def get_range_matrix():
    range_matrix = pd.DataFrame(
        [[0.00017171875, 0.005, 1, 0.001, 0.1, 0.4],
         [0.00060, 0.0194828, 3.6, 0.040, 0.4, 0.9]],
        columns=["tau", "epsilonbar", "kappa", "delta", "theta0", "alpha"]
    )
    return range_matrix

def add_j_aniso(sampling_j_aniso, ini_sampling):
    ini_sampling['j_aniso'] = sampling_j_aniso
    return ini_sampling

def OLHS_PhaseField(n_points=70, n_pop=10, max_iter=100, tol=1e-5, distance_method='euclidean', p=50, R=5, data_path = None, method = "classic"):
    n_features = 6

    sampling_j_aniso = get_sampling_j_aniso(n_points)
    range_matrix = get_range_matrix()

    ini_sampling = find_OLHS_PSO(range_matrix, n_points, n_pop, max_iter, tol, distance_method, p, R)
    ini_sampling = add_j_aniso(sampling_j_aniso, ini_sampling)
    
    ini_sampling = ini_sampling.reset_index().rename(columns={'index': 'exp_id'})
    
    if data_path != None:
        
        name =  data_path + "/doe_" + method + "_" + str(n_points) + "_points_" + datetime.now().strftime("%Y-%m-%d") + ".csv"
        
        ini_sampling.to_csv(name)
    
    return ini_sampling, name
