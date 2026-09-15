# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 10:23:32 2025

@author: egarate
"""

import pandas as pd
import numpy as np
from src.auxiliars.aux_LHS_distances import *

def scale_points(p_most_uncertain, uncertainty_df):
    
    range_matrix = pd.DataFrame(
    [[0.00017171875, 0.005, 1, 0.001, 0.1, 0.4, 4],
     [0.00060,       0.0194828, 3.6, 0.040, 0.4, 0.9, 6]],
    columns=["tau", "epsilonbar", "kappa", "delta", "theta0", "alpha", "j_aniso"])
    
    min_vals = range_matrix.iloc[0]
    max_vals = range_matrix.iloc[1]
    
    p_most_uncertain_scaled = (p_most_uncertain.iloc[:, 0:7] - min_vals) / (max_vals - min_vals)
    uncertainty_df_scaled = (uncertainty_df.iloc[:, 0:7] - min_vals) / (max_vals - min_vals)
    
    return p_most_uncertain_scaled, uncertainty_df_scaled


def get_most_close_points(p_most_uncertain, uncertainty_df):
    
    p_most_uncertain_scaled, uncertainty_df_scaled = scale_points(p_most_uncertain, uncertainty_df)
    
    max_radius = []
    
    for p_index, point in p_most_uncertain_scaled.iterrows():
        
        same_point_mask = (uncertainty_df_scaled == point).all(axis=1)
        
        uncertainty_df_scaled_point_filtered = uncertainty_df_scaled[~same_point_mask]
        
        distances_point = uncertainty_df_scaled_point_filtered.apply(lambda r: euclidean(point, r), 1)
        
        nearest_idx = distances_point.idxmin()
        
        s_max_point = distances_point[nearest_idx]/2
        
        max_radius.append(s_max_point)
        
    p_most_uncertain['max_radius'] = max_radius
    
    return p_most_uncertain

def get_mixed_uncertainty(uncertainty_df, weights = {"liquid" : 0.1, "interface" : 0.6, "solid" : 0.3}):
    
    uncertainty_df['mixed_uncertainty'] = weights['liquid']*uncertainty_df['mean_uncertainty_liquid'] + weights['interface']*uncertainty_df['mean_uncertainty_interface'] + weights['solid']*uncertainty_df['mean_uncertainty_solid']
    
    return uncertainty_df


def get_new_p_points(p_most_uncertain):
    
    range_matrix = pd.DataFrame(
    [[0.00017171875, 0.005, 1, 0.001, 0.1, 0.4, 4],
     [0.00060,       0.0194828, 3.6, 0.040, 0.4, 0.9, 6]],
    columns=["tau", "epsilonbar", "kappa", "delta", "theta0", "alpha", "j_aniso"])
    
    min_vals = range_matrix.iloc[0]
    max_vals = range_matrix.iloc[1]
    n_features  = 7
    
    generated_points = []

    for point_idx, point in p_most_uncertain.iterrows():
        
        center = point.iloc[0:7].values
        radius = point['max_radius']

        # 1. Generate aleactoric normalized direction
        direction = np.random.normal(size=n_features)
        direction /= np.linalg.norm(direction)

        # 2. Generate aleatoric distance in the hypersphere
        u = np.random.uniform(0, 1)
        scaled_radius = radius * (u * (1 / n_features))

        # 3. Generated new point
        new_point_scaled = center + direction * scaled_radius

        # 4. Verifying if is in the hypershere
        new_point_scaled = np.clip(new_point_scaled, 0, 1)

        # 5. Get the point in the original scale
        new_point_original = new_point_scaled * (max_vals.values - min_vals.values) + min_vals.values
        
        new_point_original[-1] = point['j_aniso']

        generated_points.append(new_point_original)

    # Get new points df
    df_generated = pd.DataFrame(generated_points, columns=["tau", "epsilonbar", "kappa", "delta", "theta0", "alpha", "j_aniso"])
    
    return df_generated


def main_get_next_p_points_to_sampling(p, uncertainty_df, doe):
    
    uncertainty_df =  get_mixed_uncertainty(uncertainty_df)
    
    uncertainty_sorted = uncertainty_df.sort_values(by='mixed_uncertainty', ascending=False)
    
    p_most_uncertain = uncertainty_sorted.head(p)
    
    p_most_uncertain =  get_most_close_points(p_most_uncertain, uncertainty_df)
    
    new_p_samples = get_new_p_points(p_most_uncertain)
    
    exp_new_ids = doe.iloc[-1, 0] + np.array(range(1, p + 1))
    
    new_p_samples['exp_id'] = exp_new_ids
    
    new_p_samples = new_p_samples[doe.columns]
    
    new_doe_all = pd.concat([doe, new_p_samples], ignore_index=True)
    
    return new_p_samples, new_doe_all
    
    
    
    
    
    
    
    
    
    