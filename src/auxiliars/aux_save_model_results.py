# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:27:33 2024

@author: egarate
"""
import pickle
import time

def save_model_results(results, ti=None):
    
    model_saving_path = results['model_path']
    results_saving_path = results['results_saving_path']
    
    if ti is None:
    
        model_name = model_saving_path + "/" + results['model_type'] + "_" + time.strftime("%Y-%m-%d", time.localtime()) + "_n_" + str(results["n_points"]) + ".pkl"
            
        scaler_name = model_saving_path + "/" + results['model_type'] + "_scaler_" + time.strftime("%Y-%m-%d", time.localtime()) + "_n_" + str(results["n_points"]) + ".pkl"
        
        results_name = results_saving_path  + "/" + results['model_type'] + "_results_" + time.strftime("%Y-%m-%d", time.localtime()) + "_n_" + str(results["n_points"]) + ".pkl"
        
    else:
        
        model_name = model_saving_path + "/" + results['model_type'] + "_" + time.strftime("%Y-%m-%d", time.localtime()) + "_n_" + str(results["n_points"]) + ti + ".pkl"
        
        scaler_name = model_saving_path + "/" + results['model_type'] + "_scaler_" + time.strftime("%Y-%m-%d", time.localtime()) + "_n_" + str(results["n_points"]) + ti + ".pkl"
        
        results_name = results_saving_path  + "/" + results['model_type'] + "_results_" + time.strftime("%Y-%m-%d", time.localtime()) + "_n_" + str(results["n_points"]) + ti + ".pkl"
        
    results_filtrado = {key: value for key, value in results.items() if key not in ['model', 'results_saving_path', 'model_path']}
    
    
    with open(model_name, 'wb') as file:
        pickle.dump(results['model'], file)  
    
    with open(scaler_name, 'wb') as file:
        pickle.dump(results['scaler'], file) 
        
    with open(results_name, 'wb') as file:
        pickle.dump(results_filtrado, file) 
        
    return True
        
        

def save_model_results_adaptive(model, results, ti=None):
    
    model_saving_path = results['model_path']
    results_saving_path = results['results_saving_path']
    n_samples = results['n_samples_list'][-1]
    
    if ti is None:
    
        model_name = model_saving_path + "/" + results['model_type'] + "_" + time.strftime("%Y-%m-%d", time.localtime()) + "_n_" + str(n_samples) + ".pkl"
            
        scaler_name = model_saving_path + "/" + results['model_type'] + "_scaler_" + time.strftime("%Y-%m-%d", time.localtime()) + "_n_" + str(n_samples) + ".pkl"
        
        results_name = results_saving_path  + "/" + results['model_type'] + "_results_" + time.strftime("%Y-%m-%d", time.localtime()) + "_n_" + str(results['n_points_ini']) + ".pkl"
        
    else:
        
        model_name = model_saving_path + "/" + results['model_type'] + "_" + time.strftime("%Y-%m-%d", time.localtime()) + "_n_" + str(n_samples) + ti + ".pkl"
        
        scaler_name = model_saving_path + "/" + results['model_type'] + "_scaler_" + time.strftime("%Y-%m-%d", time.localtime()) + "_n_" + str(n_samples) + ti + ".pkl"
        
        results_name = results_saving_path  + "/" + results['model_type'] + "_results_" + time.strftime("%Y-%m-%d", time.localtime()) + "_n_" + str(results['n_points_ini']) + ti + ".pkl"
        
    results['model'].append(model_name)
    results_filtrado = {key: value for key, value in results.items() if key not in ['results_saving_path', 'model_path']}
    
    if results['model_type'] != "XGB_probabilistic":
        
        model.save(model_name)
    else:    
        with open(model_name, 'wb') as file:
            pickle.dump(model, file)  
        
    with open(results_name, 'wb') as file:
        pickle.dump(results_filtrado, file) 
        
    return True
                
        
        