# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:33:54 2024

@author: egarate
"""

import pandas as pd
import numpy as np
from src.auxiliars.aux_simetry_simplification import *
from src.auxiliars.aux_feature_extraction import *
from src.auxiliars.aux_feature_selection import *
from src.auxiliars.aux_train_XGB import *
from src.auxiliars.aux_save_model_results import *
from src.auxiliars.aux_train_CNN_ini import *
from src.auxiliars.aux_train_CNN_self_supervised import *
from src.green_computing.codecarbon_emission_tracking import EmissionTrackerManager
from src.auxiliars.aux_train_MC_dropout_CNN import *
from src.auxiliars.aux_train_probabilistic_XGB import *
from src.auxiliars.aux_train_MC_dropout_CNN_self_s import *


def train_surrogate_model(neighs_t, data_raw_path, simulations, FE_method, FS_method, model, symmetry, results, ti=None, incremental=False, last_model_path = None):
        
    tracker_manager = EmissionTrackerManager()
    tracker_name = model
    if ti is not None:
        tracker_name = tracker_name + "_t_" + ti
    tracker_manager.start_tracking(function_name=tracker_name)

    if model == "XGB":
        results, ti = train_XGB_surrogate(neighs_t, data_raw_path, simulations, FE_method, FS_method, model, symmetry, results, ti)
        uncertainties_cv_df = None
        
        tracker_manager.stop_tracking(results=results)
        save_model_results(results, ti)

    elif model == "CNN_base":
        results = train_CNN_base_surrogate(neighs_t, data_raw_path, simulations, results, ti)
        uncertainties_cv_df = None
        
        tracker_manager.stop_tracking(results=results)
        save_model_results(results, ti)

    elif model == "CNN_self_supervised":
        results = train_CNN_self_supervised_surrogate(neighs_t, data_raw_path, simulations, results, ti)
        uncertainties_cv_df = None
        
        tracker_manager.stop_tracking(results=results)
        save_model_results(results, ti)

    elif model == "CNN_MC_dropout":
        model_CNN_final, results, uncertainties_cv_df = train_CNN_MC_dropout_surrogate(neighs_t, data_raw_path, simulations, results, ti, incremental, last_model_path)
        
        tracker_manager.stop_tracking(results=results)
        save_model_results_adaptive(model_CNN_final, results, ti)
        
    elif model == "XGB_probabilistic":
        results, model, uncertainties_cv_df = train_XGB_probabilistic(neighs_t, data_raw_path, simulations, FE_method, FS_method, model, symmetry, results, ti)
        
        tracker_manager.stop_tracking(results=results)
        save_model_results_adaptive(model, results, ti)
        
    elif model == "CNN_MC_dropout_ss":
        model, results, uncertainties_cv_df = train_CNN_MC_dropout_ss_surrogate(neighs_t, data_raw_path, simulations, results, ti, incremental, last_model_path)
        
        tracker_manager.stop_tracking(results=results)
        save_model_results_adaptive(model, results, ti)
        

    return results, uncertainties_cv_df

def train_XGB_surrogate(neighs_t, data_raw_path, simulations, FE_method="radius_method", FS_method=None, model="XGB", symmetry=None, results=None, ti=None):
    
    data_tabular, t_FE = feature_extraction(neighs_t, data_raw_path, simulations, FE_method)
    
    results["comp_costs"]["t_FE"] = t_FE
    

    data_tabular, t_FS = feature_selection(data_tabular, FS_method)
    results["comp_costs"]["t_FS"] = t_FS


    if symmetry != None:
        
        data_to_train, t_symmetry = divide_image(data_experiments=data_tabular, division_dict=symmetry)
        
        del data_tabular
        
    else:
        t_symmetry = 0
        data_to_train = data_tabular
        
    results["comp_costs"]["t_symmetry"] = t_symmetry

    
    results = train_xgb(data_to_train, results)
    
    
    return results, ti

def train_CNN_base_surrogate(neighs_t, data_raw_path, simulations, results=None, ti=None):
    
    model_hyperparameters = results['hyperparameters']
    
    X_images_temp, X_images_op, X_tabular, Y_output, exp_ids = load_data(data_raw_path, simulations, results['t_instances'])
    
    model_CNN_final, r2_mean, r2_all,  y_pred, t_train = train_CNN_baseline_model(X_images_temp = X_images_temp, X_images_op = X_images_op, 
                                                                         X_tabular=X_tabular, Y_output=Y_output,
                                                                         exp_ids=exp_ids, epochs=model_hyperparameters['epochs'], 
                                                                         batch_size=model_hyperparameters['batch_size'], 
                                                                         validation_split=model_hyperparameters['validation_split'], 
                                                                         Nx=100, Ny=100, t_instances=results['t_instances'], 
                                                                         hyperparameters_images = model_hyperparameters['CNN'], 
                                                                         hyperparameters_tabular =  model_hyperparameters['tabular'], 
                                                                         model_optimizer = model_hyperparameters['optimizer'], 
                                                                         model_loss = results["loss_function"], n_cv = 10)
    
    results['model'] = model_CNN_final
    results['model_metrics']['cv_metric'] = r2_mean
    results["comp_costs"]["t_training"] = t_train
    
    results['model_metrics']['global_metric'] = r2_all

    
    return results
    
def train_CNN_MC_dropout_surrogate(neighs_t, data_raw_path, simulations, results=None, ti=None, incremental = False, last_model_path = None):
    
    model_hyperparameters = results['model_hyperparameters']
    
    X_images_temp, X_images_op, X_tabular, Y_output, exp_ids = load_data(data_raw_path, simulations, results['t_instances'])
    
    model_CNN_final, r2_mean, r2_all, uncertainties_cv_df, y_pred, t_train = train_CNN_MC_dropout(X_images_temp = X_images_temp, X_images_op = X_images_op, 
                                                                         X_tabular=X_tabular, Y_output=Y_output,
                                                                         exp_ids=exp_ids, epochs=model_hyperparameters['epochs'], 
                                                                         batch_size=model_hyperparameters['batch_size'], 
                                                                         validation_split=model_hyperparameters['validation_split'], 
                                                                         Nx=100, Ny=100, t_instances=results['t_instances'], 
                                                                         hyperparameters_images = model_hyperparameters['CNN'], 
                                                                         hyperparameters_tabular =  model_hyperparameters['tabular'], 
                                                                         model_optimizer = model_hyperparameters['optimizer'], 
                                                                         model_loss = results["loss_function"], n_cv = 10, 
                                                                         incremental = incremental, last_model_path = last_model_path)
    
    results['model_metrics']['cv_metric'].append(r2_mean)
    results["comp_costs"]["t_training"].append(t_train)
    
    return model_CNN_final, results, uncertainties_cv_df
    
def train_CNN_self_supervised_surrogate(neighs_t, data_raw_path, simulations, results=None, ti=None):
    
    model_hyperparameters = results['hyperparameters']
    
    X_images_temp, X_images_op, X_tabular, Y_output, exp_ids = load_data(data_raw_path, simulations, results['t_instances'])
    
    model_CNN_final, r2_mean, r2_all, y_pred, t_train = train_CNN_self_supervised_model(X_images_temp = X_images_temp, X_images_op = X_images_op, 
                                                                         X_tabular=X_tabular, Y_output=Y_output,
                                                                         exp_ids=exp_ids, epochs=model_hyperparameters['epochs'], 
                                                                         batch_size=model_hyperparameters['batch_size'], 
                                                                         validation_split=model_hyperparameters['validation_split'], 
                                                                         Nx=100, Ny=100, t_instances=results['t_instances'], 
                                                                         hyperparameters_images = model_hyperparameters['CNN'], 
                                                                         hyperparameters_tabular =  model_hyperparameters['tabular'], 
                                                                         model_optimizer = model_hyperparameters['optimizer'], 
                                                                         model_loss = results["loss_function"], n_cv = 10)
    
    results['model'] = model_CNN_final
    results['model_metrics']['cv_metric'] = r2_mean
    results["comp_costs"]["t_training"] = t_train
    
    results['model_metrics']['global_metric'].append(r2_all)
    
    return results    
    
def train_XGB_probabilistic(neighs_t, data_raw_path, simulations, FE_method="radius_method", FS_method=None, model="XGB", symmetry=None, results=None, ti=None):

    data_tabular, t_FE = feature_extraction(neighs_t, data_raw_path, simulations, FE_method)
    
    results["comp_costs"]["t_FE"].append(t_FE)
    

    data_tabular, t_FS = feature_selection(data_tabular, FS_method)
    results["comp_costs"]["t_FS"].append(t_FS)


    if symmetry != None:
        
        data_to_train, t_symmetry = divide_image(data_experiments=data_tabular, division_dict=symmetry)
        
        del data_tabular
        
    else:
        t_symmetry = 0
        data_to_train = data_tabular
        
    results["comp_costs"]["t_symmetry"].append(t_symmetry)
    
    results, model, uncertainty_cv_values = train_xgb_probabilistic(data_to_train, results, n_CV=10)
    
    return results, model, uncertainty_cv_values
    
def train_CNN_MC_dropout_ss_surrogate(neighs_t, data_raw_path, simulations, results, ti, incremental, last_model_path):
    
    model_hyperparameters = results['model_hyperparameters']
    
    X_images_temp, X_images_op, X_tabular, Y_output, exp_ids = load_data(data_raw_path, simulations, results['t_instances'])
    
    model_CNN_final, r2_mean, r2_all, uncertainties_cv_df, y_pred, t_train = train_CNN_MC_dropout_self_supervised(X_images_temp = X_images_temp, X_images_op = X_images_op, 
                                                                         X_tabular=X_tabular, Y_output=Y_output,
                                                                         exp_ids=exp_ids, epochs=model_hyperparameters['epochs'], 
                                                                         batch_size=model_hyperparameters['batch_size'], 
                                                                         validation_split=model_hyperparameters['validation_split'], 
                                                                         Nx=100, Ny=100, t_instances=results['t_instances'], 
                                                                         hyperparameters_images = model_hyperparameters['CNN'], 
                                                                         hyperparameters_tabular =  model_hyperparameters['tabular'], 
                                                                         model_optimizer = model_hyperparameters['optimizer'], 
                                                                         model_loss = results["loss_function"], n_cv = 10,
                                                                         incremental = incremental, last_model_path = last_model_path)
    
    results['model_metrics']['cv_metric'].append(r2_mean)
    results["comp_costs"]["t_training"].append(t_train)
    
    return model_CNN_final, results, uncertainties_cv_df
    
    