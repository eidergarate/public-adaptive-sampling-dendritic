# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:29:37 2024

@author: egarate
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def y1_division(x_i):
    
    return np.ceil(1/3*x_i + 100/3)

def y2_division(x_i):
    
    return np.ceil(-1/3*x_i + 200/3)

def divide_image(data_experiments, division_dict):
    
    t_0 = time.time()
    
    data_4 = data_experiments[data_experiments['j_aniso'] == 4]
    data_6 = data_experiments[data_experiments['j_aniso'] == 6]
    
    if division_dict['aniso4'] == 4:
        
        filtered_data_4 = data_4[(data_4['x'] >= 0) & (data_4['x'] <= 50) & (data_4['y'] >= 0) & (data_4['y'] <= 50)]
    
    elif division_dict['aniso4'] == 2:
        
        filtered_data_4 = data_4[(data_4['x'] >= 0) & (data_4['x'] <= 50) & (data_4['y'] >= 0) & (data_4['y'] <= 100)]
        
        
    
    if division_dict['aniso6'] == 6:
        
        data_6['y1'] = data_6['x'].apply(y1_division)
        filtered_data_6 = data_6[(data_6['x'] >= 0) & (data_6['x'] <= 50) & (data_6['y'] >= 0) & (data_6['y'] <= data_6['y1'])]
        filtered_data_6.drop(['y1'], axis=1, inplace=True)
    
    elif division_dict['aniso6'] == 2:
        
        filtered_data_6 = data_6[(data_6['x'] >= 0) & (data_6['x'] <= 50) & (data_6['y'] >= 0) & (data_6['y'] <= 100)]


    filtered_data_experiments = pd.concat([filtered_data_4, filtered_data_6])
    
    t_symmetry = time.time() - t_0
    
    return filtered_data_experiments, t_symmetry


def rotate_points(x, y, angle_degrees, center=(50, 50)):

    angle_radians = np.deg2rad(angle_degrees)
    x_shifted = x - center[0]
    y_shifted = y - center[1]
    
    x_rotated = x_shifted * np.cos(angle_radians) - y_shifted * np.sin(angle_radians) + center[0]
    y_rotated = x_shifted * np.sin(angle_radians) + y_shifted * np.cos(angle_radians) + center[1]
    
    return np.round(x_rotated).astype(int), np.round(y_rotated).astype(int)

def reconstruct_image_6parts(filtered_data_experiments, exp_id):
    
    A1 = filtered_data_experiments[filtered_data_experiments['exp_id'] == exp_id]
    
    all_regions = []
    
    all_regions.append(A1)

    angles = [60, 120, 180, 240, 300] 
    
    for angle in angles:
        
        x_rotated, y_rotated = rotate_points(A1['x'], A1['y'], angle)
        
        rotated_region = pd.DataFrame({
            'x': x_rotated,
            'y': y_rotated,
            'OP_4000': A1['OP_4000'],  
            'exp_id': exp_id
        })
        all_regions.append(rotated_region)
    
    reconstructed_image = pd.concat(all_regions, ignore_index=True)
    
    return reconstructed_image

def reconstruct_image_2parts(filtered_data_experiments, exp_id):
    
    A1 = filtered_data_experiments[filtered_data_experiments['exp_id'] == exp_id]
    
    all_regions = []
    
    all_regions.append(A1)

    angles = [180] 
    
    for angle in angles:
        
        x_rotated, y_rotated = rotate_points(A1['x'], A1['y'], angle)
        
        rotated_region = pd.DataFrame({
            'x': x_rotated,
            'y': y_rotated,
            'OP_4000': A1['OP_4000'],  
            'exp_id': exp_id
        })
        all_regions.append(rotated_region)
    
    reconstructed_image = pd.concat(all_regions, ignore_index=True)
    
    return reconstructed_image

def reconstruct_image_4parts(filtered_data_experiments, exp_id):
    
    A1 = filtered_data_experiments[filtered_data_experiments['exp_id'] == exp_id]
    
    all_regions = []
    
    all_regions.append(A1)

    angles = [90, 180, 270] 
    
    for angle in angles:
        
        x_rotated, y_rotated = rotate_points(A1['x'], A1['y'], angle)
        
        rotated_region = pd.DataFrame({
            'x': x_rotated,
            'y': y_rotated,
            'OP_4000': A1['OP_4000'],  
            'exp_id': exp_id
        })
        all_regions.append(rotated_region)
    
    reconstructed_image = pd.concat(all_regions, ignore_index=True)
    
    return reconstructed_image


def reconstruct_all_images(filtered_data_experiments_OP, division_dict):
    reconstructed_images = []

    data_4 = filtered_data_experiments_OP[filtered_data_experiments_OP['j_aniso'] == 4]
    data_6 = filtered_data_experiments_OP[filtered_data_experiments_OP['j_aniso'] == 6]

    if division_dict['aniso4'] == 4:
        
        exp_ids_4 = data_4['exp_id'].unique()
        
        for exp_id in exp_ids_4:
            
            reconstructed_image = reconstruct_image_4parts(data_4, exp_id)
            reconstructed_images.append(reconstructed_image)
            

    elif division_dict['aniso4'] == 2:
        
        exp_ids_4 = data_4['exp_id'].unique()
        
        for exp_id in exp_ids_4:
            
            reconstructed_image = reconstruct_image_2parts(data_4, exp_id)
            reconstructed_images.append(reconstructed_image)
            
            

    if division_dict['aniso6'] == 6:
        
        exp_ids_6 = data_6['exp_id'].unique()
        
        for exp_id in exp_ids_6:
            
            reconstructed_image = reconstruct_image_6parts(data_6, exp_id)
            reconstructed_images.append(reconstructed_image)
            

    elif division_dict['aniso6'] == 2:
        
        exp_ids_6 = data_6['exp_id'].unique()
        
        for exp_id in exp_ids_6:
            
            reconstructed_image = reconstruct_image_2parts(data_6, exp_id)
            reconstructed_images.append(reconstructed_image)
            
            
            

    full_reconstructed_df = pd.concat(reconstructed_images, ignore_index=True)

    return full_reconstructed_df


