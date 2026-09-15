# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:40:38 2024

@author: egarate
"""

import numpy as np
import pandas as pd
import os
from src.auxiliars.aux_cleaning_and_rename import *
from src.auxiliars.aux_edge_pixels_inputation import *


def get_neighbour_names(r):
    final_colnames = []

    for i in range(-r, r + 1):
        row = []
        for j in range(-r, r + 1):
            if i == 0 and j == 0:
                row.append("(xi, yi)")  
            else:
                direction = ""
                if i < 0:
                    direction += f"up{abs(i)}"
                elif i > 0:
                    direction += f"down{i}"
                
                if j < 0:
                    direction += f"left{abs(j)}"
                elif j > 0:
                    direction += f"right{j}"
                
                row.append(direction)
        final_colnames.extend(row)
    
    return {"T_names": final_colnames, "OP_names": final_colnames}




def get_x_i_y_i_neighbours(op_data, temp_data, x_i, y_i, r):
   
    op_data = np.array(op_data)
    temp_data = np.array(temp_data)

    
    from_x = max(0, x_i - r - 1)
    to_x = min(x_i + r, op_data.shape[0])
    
    from_y = max(0, y_i - r - 1)
    to_y = min(y_i + r, op_data.shape[1])

    op_neighbours = op_data[from_x:to_x, from_y:to_y]
    temp_neighbours = temp_data[from_x:to_x, from_y:to_y]

    
    op_neighbours, temp_neighbours = _pad_matrix_to_5x5(op_neighbours, temp_neighbours, x_i, y_i, r, op_data.shape, temp_data.shape)

    op_neighbours = op_neighbours.flatten(order='C')  
    temp_neighbours = temp_neighbours.flatten(order='C')

    return {"op_neighbours": op_neighbours, "temp_neighbours": temp_neighbours}

def _pad_matrix_to_5x5(op_neighbours, temp_neighbours, x_i, y_i, r, op_shape, temp_shape):

    op_padded = np.full((2 * r + 1, 2 * r + 1), np.nan)
    temp_padded = np.full((2 * r + 1, 2 * r + 1), np.nan)


    if x_i <= r:  
        start_x = r - (x_i - 1)
    elif x_i >= op_shape[0] - r: 
        start_x = 0
    else:  
        start_x = r - r  

    end_x = start_x + op_neighbours.shape[0]


    if y_i <= r:  
        start_y = r - (y_i - 1)
    elif y_i >= op_shape[1] - r:  
        start_y = 0
    else:  
        start_y = r - r  

    end_y = start_y + op_neighbours.shape[1]


    op_padded[start_x:end_x, start_y:end_y] = op_neighbours
    temp_padded[start_x:end_x, start_y:end_y] = temp_neighbours

    return op_padded, temp_padded





def get_xi_yi_obs(op_data, temp_data, x_i, y_i, r):
    neighbours = get_x_i_y_i_neighbours(op_data, temp_data, x_i, y_i, r)
    op_neighbours = neighbours["op_neighbours"]
    temp_neighbours = neighbours["temp_neighbours"]

    xi_yi_obs = np.concatenate(([x_i, y_i], op_neighbours, temp_neighbours))
    
    return xi_yi_obs


def construct_data_per_exp(exp_row, r=2, t_step="4000", data_raw = None):
    
    ref_path = data_raw + "/" + f"Ref_{int(exp_row[0])}"
    ref_op_csv = ref_path + "/Ref_" + str(int(exp_row[0])) + "_op_" + str(int(t_step)) + ".csv"
    ref_temp_csv = ref_path + "/Ref_" + str(int(exp_row[0])) + "_temp_" + str(int(t_step)) + ".csv"

    op_data = pd.read_csv(ref_op_csv, header=None)
    temp_data = pd.read_csv(ref_temp_csv, header=None)

    df_colnames = get_neighbour_names(r)

    exp_df_list = []
    for idx_long in range(op_data.shape[0] * op_data.shape[1]):
        x_i = idx_long // op_data.shape[1] + 1
        y_i = idx_long % op_data.shape[1] + 1

        new_obs = pd.DataFrame([get_xi_yi_obs(op_data, temp_data, x_i, y_i, r)])

        new_obs.columns = ["x_pos", "y_pos"] + \
                          [f"OP_{name}_{t_step}" for name in df_colnames["OP_names"]] + \
                          [f"T_{name}_{t_step}" for name in df_colnames["T_names"]]
        exp_df_list.append(new_obs)

    exp_df = pd.concat(exp_df_list, ignore_index=True)

    for col, value in zip(["tau", "epsilonbar", "kappa", "delta", "theta0", "alpha", "j_aniso"], exp_row[1:]):
        exp_df[col] = value

    return exp_df

def constructDataFrameAllExp_lastT(r):
    
    df_with_exp_ids = pd.read_excel(os.path.join("physical-model", "datos", "parametrosSimulaciones.xlsx"))
    df_with_exp_ids = df_with_exp_ids.dropna()  
    
    df_all_experiments = pd.concat([construct_data_per_exp(row, r) for _, row in df_with_exp_ids.iterrows()], ignore_index=True)
    
    return df_all_experiments



def constructSpatioTemporal_all(r, neighs_t, parametrosSimulaciones=None, data_path=None, data_raw = None):
    
    lastT = 4000
    if parametrosSimulaciones is None:
        parametrosSimulaciones = pd.read_csv(data_path, sep=",", dtype=str)
        
    all_data_list = []
    for _, exp_row in parametrosSimulaciones.iterrows():
        exp_df_list = []
        exp_id = int(exp_row[0])
        for t_step in neighs_t:
            
            exp_df = construct_data_per_exp(exp_row, r, t_step=str(t_step), data_raw=data_raw)
            exp_df_list.append(exp_df)
        
       
        data_exp = exp_df_list[0]
        for df in exp_df_list[1:]:
            data_exp = pd.merge(data_exp, df, on=["x_pos", "y_pos", "tau", "epsilonbar", "kappa", "delta", "j_aniso", "alpha", "theta0"], how='inner')
        
        data_exp['exp_id'] = exp_id
        
        
        all_data_list.append(data_exp)
        
    data_all = pd.concat(all_data_list, ignore_index = True)
              
    data_all_cleaned = cleaning_and_rename(data_all)
    data_all_inputed = imput_df_nearest(data_all_cleaned, t_instances = neighs_t)
    
    return data_all_inputed
