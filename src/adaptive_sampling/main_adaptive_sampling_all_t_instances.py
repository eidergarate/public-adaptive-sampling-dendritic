# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 09:04:31 2025

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

def initialize_time_instance():
    
    t3 = list(range(1000, 2500, 500))
    t3.append(4000)
    
    t6 = list(range(1000, 3000, 500))
    t6.append(4000)
    
    t9 = list(range(1000, 3500, 500))
    t9.append(4000)
    
    t_instances_list = {
        "t3" : t3,
        "t6" : t6,
        "t9" : t9,
       }
    
    
    return t_instances_list

def main_adaptive_sampling_and_modeling_all_t_instances(n_ini_points, p_adding_points, n_samples_max, 
                                        FE_method="radius_method", FS_method=None,
                                         model="XGB", symmetry=None, loss_function="MSE", save_dataset=True, 
                                         model_metrics=None, model_hyperparameters=None,
                                         saving_paths={'doe':'C:/Users/egarate/Desktop/Tesis_Adaptive_Sampling/data'},
                                         incremental = True, n_retrain_all = 50, retrain_some = False):
    
    tracker_manager = EmissionTrackerManager()

    t_instances_dict = initialize_time_instance()
    
    if model == "XGB_probabilistic":
        
        t_instances_dict = {k: t_instances_dict[k] for k in ['t6', 't9']}
        
    if model == "CNN_MC_dropout_ss":
        t_instances_dict = {k: t_instances_dict[k] for k in ['t6', 't9']}

    
    t_for_simulation =  list(range(200, 3100, 100))
    t_for_simulation.append(4000)

    initial_CO2_results = {"CO2_emissions_kg": {}}
    tracker_manager.start_tracking(function_name="create_sampling")

    t_doe_ini = time.time()
    doe_ini, doe_path = OLHS_PhaseField(n_ini_points, n_pop=100, max_iter=100, tol=1e-7, distance_method='euclidean', p=50, R=0.7, data_path=saving_paths['doe'], method = "adaptive")
    t_doe = time.time() - t_doe_ini
    
    tracker_manager.stop_tracking(results=initial_CO2_results)
    
    tracker_manager.start_tracking(function_name=f"simulate_experiments") 

    t_simulation_ini = time.time()
    simulate_phase_field_experiments(doe_path=doe_path, data_raw_path=saving_paths['data_raw'], idxToSave=t_for_simulation)
    t_simulation = time.time() - t_simulation_ini
    
    tracker_manager.stop_tracking(results=initial_CO2_results)
    
    for t_instances_experiment in t_instances_dict.keys():
        
        p_adding_points_iter = int(np.round(n_ini_points*p_adding_points))
        
        t_instances = t_instances_dict[t_instances_experiment]
        results = get_results_dict(n_ini_points, p_adding_points, n_samples_max, t_instances, FE_method, FS_method, model, symmetry, loss_function, model_metrics, saving_paths, model_hyperparameters, n_retrain_all, retrain_some)
        
        results["comp_costs"]["t_doe"] = []
        results["comp_costs"]["t_doe"].append(t_doe)
    
        results["comp_costs"]["t_simulation"] = []
        results["comp_costs"]["t_simulation"].append(t_simulation)
        results["comp_costs"]["t_training"] = []
        results["comp_costs"]["t_FE"] = []
        results["comp_costs"]["t_FS"] = []
        results["comp_costs"]["t_symmetry"] = []
    
        n_samples = n_ini_points
        last_model_path = False
        ti = t_instances_experiment
        doe = doe_ini
        
        while n_samples <=  n_samples_max:
           
            results['n_samples_list'].append(n_samples)
        
            if model != "XGB_probabilistic":
            
                if ((n_samples - n_ini_points) % n_retrain_all == 0) and retrain_some:
                
                    t_retrain_0 = time.time()    
                    results, uncertainties_cv_df = train_surrogate_model(neighs_t=t_instances, data_raw_path = saving_paths['data_raw'], simulations = doe, FE_method = FE_method, FS_method = FS_method, model = model, symmetry = symmetry, results = results, ti=ti, incremental = False, last_model_path = last_model_path)
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
        
            if n_samples < n_samples_max:
                tracker_manager.start_tracking(function_name=f"create_sampling_t_{ti}")

                t_doe_adaptive_0 = time.time()
                new_p_samples, new_doe_all = main_get_next_p_points_to_sampling(p_adding_points_iter, uncertainties_cv_df, doe)
                t_doe_adaptive = time.time() - t_doe_adaptive_0
                results["comp_costs"]["t_doe"].append(t_doe_adaptive)

                tracker_manager.stop_tracking(results=results)
    
                tracker_manager.start_tracking(function_name=f"simulate_experiments_t_{ti}") 
                
                t_sim_0 = time.time()
                simulate_phase_field_adaptive(new_added_points=new_p_samples, doe_path=doe_path, data_raw_path=saving_paths['data_raw'], idxToSave=t_instances)
                t_sim_adaptive = time.time() - t_sim_0
                results["comp_costs"]["t_simulation"].append(t_sim_adaptive)

                tracker_manager.stop_tracking(results=results)

                doe_path_new = saving_paths['doe'] + "/doe_adative_" + str(n_ini_points) + "_points_"  + ti + "_" + model + "_" + datetime.now().strftime("%Y-%m-%d") + ".csv"
                new_doe_all.to_csv(doe_path_new)
        
                doe = new_doe_all
                
                n_samples = n_samples + p_adding_points_iter
                
                p_adding_points_iter = int(np.round(n_samples*p_adding_points))
              
                last_model_path = results['model'][-1]
            
            if n_samples == n_samples_max:
                break
    