# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:26:34 2024

@author: egarate
"""

import pandas as pd
import numpy as pd
from src.Optimal_LHS.OLHS_phase_field import *
from src.auxiliars.aux_simulate_samples import *
from src.auxiliars.aux_surrogate_modeling_training import *
import time


def get_results_dict(n_points, t_instances, FE_method, FS_method, model, symmetry, loss_function, model_metrics, saving_paths, model_hyperparameters):
  
    if model != "XGB":
        CNN_hyperparameters = model_hyperparameters["CNN"]
    
        CNN_hyperparameter_image_i = CNN_hyperparameters[0]
        CNN_hyperparameters_combined = CNN_hyperparameters[1]
    
        CNN_hyperparameters_all = []
    
        for instance in range(len(t_instances) - 1):
        
            CNN_hyperparameters_all.append(CNN_hyperparameter_image_i)
    
        CNN_hyperparameters_all.append(CNN_hyperparameters_combined)
    
        model_hyperparameters["CNN"] = CNN_hyperparameters_all
        
    
    results = {"n_points" : n_points,
               "t_instances" : t_instances,
               "t_dimension" : len(t_instances),
               "FE_method" : FE_method,
               "FS_method" : FS_method,
               "model_type" : model,
               "hyperparameters": model_hyperparameters,
               "model" : None,
               "scaler": None,
               "model_path": saving_paths['model_path'],
               "symmetry" : symmetry,
               "loss_function" : loss_function,
               "comp_costs" : {},
               "model_metrics" : model_metrics,
               "results_saving_path" : saving_paths['results_saving_path'],
               "CO2_emissions_kg": {}
               }
    
    return results

def initialize_time_instance():
    
    
    t1 = list(range(200, 2000, 500))
    t1.extend([2000, 4000]) 
    t2 = list(range(600, 2000, 500))
    t2.extend([2000, 4000])
    t3 = list(range(1000, 2500, 500))
    t3.append(4000)
    
    t4 = list(range(200, 2500, 500))
    t4.extend([2500, 4000]) 
    t5 = list(range(600, 2500, 500))
    t5.extend([2500, 4000])
    t6 = list(range(1000, 3000, 500))
    t6.append(4000)
    
    t7 = list(range(200, 3000, 500))
    t7.extend([3000, 4000]) 
    t8 = list(range(600, 3000, 500))
    t8.extend([3000, 4000])
    t9 = list(range(1000, 3500, 500))
    t9.append(4000)
    
    t_instances_list = {
        "t3" : t3,
        "t6" : t6,
        "t9" : t9,
       }
    
    
    return t_instances_list



def main_classical_sampling_and_modeling_all_t_instances(n_points, FE_method="radius_method", FS_method=None,
                                         model="XGB", symmetry=None, loss_function="MSE", save_dataset=True, 
                                         model_metrics=None, model_hyperparameters=None,
                                         saving_paths={'doe':'C:/Users/egarate/Desktop/Tesis_Adaptive_Sampling/data'}):
    
    tracker_manager = EmissionTrackerManager()
    t_instances_dict = initialize_time_instance()
    
    t_for_simulation =  list(range(200, 3100, 100))
    t_for_simulation.append(4000)
    
    initial_CO2_results = {"CO2_emissions_kg": {}}

    # 1. Create sampling 
    print("Starting OLHS calculation")
    tracker_manager.start_tracking(function_name="create_sampling")
    t_doe_ini = time.time()
    doe, doe_path = OLHS_PhaseField(n_points, n_pop=100, max_iter=100, tol=1e-7, distance_method='euclidean', p=50, R=0.7, data_path=saving_paths['doe'])
    t_doe = time.time() - t_doe_ini
    tracker_manager.stop_tracking(results=initial_CO2_results)
    print("OLHS obtained correctly")
    
    # 2. Simulate experiments 
    print("Simulation started")
    tracker_manager.start_tracking(function_name="simulate_experiments")
    t_simulation_ini = time.time()
    simulate_phase_field_experiments(doe_path=doe_path, data_raw_path=saving_paths['data_raw'], idxToSave=t_for_simulation)
    t_simulation = time.time() - t_simulation_ini
    tracker_manager.stop_tracking(results=initial_CO2_results)
    print("Simulation completed")
    
    if model == "CNN_base" and n_points == 500:
        t_instances_dict_new = {'t5' : t_instances_dict['t5'], 
	      't6' : t_instances_dict['t6'],
	      't7' : t_instances_dict['t7'], 
	      't8' : t_instances_dict['t8'], 
	      't9' : t_instances_dict['t9']}
          
        t_instances_dict = t_instances_dict_new
    
    for t_instances_experiment in t_instances_dict.keys():
        
        t_instances = t_instances_dict[t_instances_experiment]
        
        results = get_results_dict(n_points, t_instances, FE_method, FS_method, model, symmetry, loss_function, model_metrics, saving_paths, model_hyperparameters)
        results["CO2_emissions_kg"].update(initial_CO2_results["CO2_emissions_kg"])
        results["comp_costs"]["t_doe"] = t_doe
        results["comp_costs"]["t_simulation"] = t_simulation
        print("Training started")
        print("training with ", n_points, " samples")
        print(model, " model training")
        print("training with ", t_instances_experiment, " instances")
        results = train_surrogate_model(neighs_t=t_instances, data_raw_path = saving_paths['data_raw'], simulations = doe, FE_method = FE_method, FS_method = FS_method, model = model, symmetry = symmetry, results = results, ti = t_instances_experiment)
        print("Training completed and results saved correctly")
    
    return True










    
