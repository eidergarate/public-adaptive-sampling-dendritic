# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 13:16:03 2025

@author: egarate
"""

import pandas as pd
import numpy as pd
from src.Optimal_LHS.OLHS_phase_field import *
from src.auxiliars.aux_simulate_samples import *
from src.auxiliars.aux_surrogate_modeling_training import *
from src.green_computing.codecarbon_emission_tracking import EmissionTrackerManager
from src.auxiliars.aux_add_adaptive_samples import *
import time

#n_points: integer, number of points
#t_instances: integer list of time instances to be used
#FE_method: string, feature extraction method
#FS_method: string, feature selection method
#model: string, model type to be used
#symmetry: integer, symmetry type 2, 4, or 6
#loss_function: string, loss_function type
#model_metrics: dictionary, with metric types as keys, and lists of 2 values with 
# the metric name and metric actual value (when get it).

def get_results_dict(n_ini_points, p_adding_points, n_samples_max, t_instances, FE_method, FS_method, model, symmetry, loss_function, model_metrics, saving_paths, model_hyperparameters, n_retrain_all = 50, retrain_some = False):
  
    if model != "XGB_probabilistic":
        CNN_hyperparameters = model_hyperparameters["CNN"]
    
        CNN_hyperparameter_image_i = CNN_hyperparameters[0]
        CNN_hyperparameters_combined = CNN_hyperparameters[1]
    
        CNN_hyperparameters_all = []
    
        for instance in range(len(t_instances) - 1):
        
            CNN_hyperparameters_all.append(CNN_hyperparameter_image_i)
    
        CNN_hyperparameters_all.append(CNN_hyperparameters_combined)
    
        model_hyperparameters["CNN"] = CNN_hyperparameters_all
       
    results = {"n_points_ini" : n_ini_points, 
               "p_adding_points" : p_adding_points,
               "n_samples_max" : n_samples_max,
               "n_samples_list" : [],
               "t_instances" : t_instances,
               "t_dimension" : len(t_instances),
               "FE_method" : FE_method,
               "FS_method" : FS_method,
               "model_type" : model,
               "model_hyperparameters": model_hyperparameters,
               "model" : [], #izenak
               "scaler": [], #izenak
               "model_path": saving_paths['model_path'],
               "symmetry" : symmetry,
               "loss_function" : loss_function,
               "comp_costs" : {},
               "model_metrics" : model_metrics,
               "results_saving_path" : saving_paths['results_saving_path'],
               "CO2_emissions_kg": {},
               "incremental_all_retrain" : retrain_some,
               "all_retrain_every" : n_retrain_all
               }
    
    return results


def main_adaptive_sampling_and_modeling(n_ini_points, p_adding_points, n_samples_max, t_instances, 
                                        FE_method="radius_method", FS_method=None,
                                         model="XGB", symmetry=None, loss_function="MSE", save_dataset=True, 
                                         model_metrics=None, model_hyperparameters=None,
                                         saving_paths={'doe':'C:/Users/egarate/Desktop/Tesis_Adaptive_Sampling/data'},
                                         incremental = True, n_retrain_all = 50, retrain_some = False):
    
    tracker_manager = EmissionTrackerManager()

    results = get_results_dict(n_ini_points, p_adding_points, n_samples_max, t_instances, FE_method, FS_method, model, symmetry, loss_function, model_metrics, saving_paths, model_hyperparameters, n_retrain_all, retrain_some)
    
    tracker_manager.start_tracking(function_name="create_sampling")

    t_doe_ini = time.time()
    doe, doe_path = OLHS_PhaseField(n_ini_points, n_pop=100, max_iter=100, tol=1e-7, distance_method='euclidean', p=50, R=0.7, data_path=saving_paths['doe'], method = "adaptive")
    t_doe = time.time() - t_doe_ini
    
    results["comp_costs"]["t_doe"] = []
    results["comp_costs"]["t_doe"].append(t_doe)
    tracker_manager.stop_tracking(results=results)
    
    tracker_manager.start_tracking(function_name=f"simulate_experiments") 

    t_simulation_ini = time.time()
    simulate_phase_field_experiments(doe_path=doe_path, data_raw_path=saving_paths['data_raw'], idxToSave=t_instances)
    t_simulation = time.time() - t_simulation_ini

    tracker_manager.stop_tracking(results=results)
    
    results["comp_costs"]["t_simulation"] = []
    results["comp_costs"]["t_simulation"].append(t_simulation)
    results["comp_costs"]["t_training"] = []
    results["comp_costs"]["t_FE"] = []
    results["comp_costs"]["t_FS"] = []
    results["comp_costs"]["t_symmetry"] = []

    
    n_samples = n_ini_points
    last_model_path = False
        
    while n_samples <=  n_samples_max:
        
        ti = None
        results['n_samples_list'].append(n_samples)

        if model != "XGB_probabilistic":
            
            if ((n_samples - n_ini_points) % n_retrain_all == 0) and retrain_some:
                
                t_retrain_0 = time.time() 
                results, uncertainties_cv_df =  train_surrogate_model(neighs_t=t_instances, data_raw_path = saving_paths['data_raw'], simulations = doe, FE_method = FE_method, FS_method = FS_method, model = model, symmetry = symmetry, results = results, ti=ti, incremental = False, last_model_path = last_model_path)
                t_retrain = time.time()
                results["comp_costs"]["t_training"].append(t_retrain)
                
            else:
                
                t_retrain_0 = time.time()
                results, uncertainties_cv_df =  train_surrogate_model(neighs_t=t_instances, data_raw_path = saving_paths['data_raw'], simulations = doe, FE_method = FE_method, FS_method = FS_method, model = model, symmetry = symmetry, results = results, ti=ti, incremental = incremental, last_model_path = last_model_path)
                t_retrain = time.time()
                results["comp_costs"]["t_training"].append(t_retrain)
                
        else:
            
            t_retrain_0 = time.time()
            results, uncertainties_cv_df =  train_surrogate_model(neighs_t=t_instances, data_raw_path = saving_paths['data_raw'], simulations = doe, FE_method = FE_method, FS_method = FS_method, model = model, symmetry = symmetry, results = results, ti=ti, incremental = incremental, last_model_path = last_model_path)
            t_retrain = time.time()
            results["comp_costs"]["t_training"].append(t_retrain)
        
        if n_samples < n_samples_max: # ??
        
            new_p_samples, new_doe_all = main_get_next_p_points_to_sampling(p_adding_points, uncertainties_cv_df, doe)
            
            tracker_manager.start_tracking(function_name=f"simulate_adaptive_n-{new_p_samples}") 
            t_sim_0 = time.time()
            simulate_phase_field_adaptive(new_added_points=new_p_samples, doe_path=doe_path, data_raw_path=saving_paths['data_raw'], idxToSave=t_instances)
            t_sim_adaptive = time.time() - t_sim_0
            results["comp_costs"]["t_simulation"].append(t_sim_adaptive)
            tracker_manager.stop_tracking(results=results)
            new_doe_all.to_csv(doe_path)
        
            doe = new_doe_all
        
            n_samples = n_samples + p_adding_points
        
            last_model_path = results['model'][-1]
            
        if n_samples == n_samples_max: # ??
            break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    