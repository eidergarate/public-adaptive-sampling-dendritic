# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:02:10 2025

@author: egarate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

        

def unify_and_get_all_training_costs(results_path, files_list = None):
    
    training_results_all = []
    
    if files_list == None:
        
        for results_file in os.listdir(results_path):
        
            if results_file.endswith(".pkl"):  
                
                results_file_path = os.path.join(results_path, results_file)
            
                with open(results_file_path, "rb") as f:
                    
                    results_i = pickle.load(f)
                    
                model_type =  results_i["model_type"]
                t_dimension = results_i["t_dimension"]
                t_instances = results_i["t_instances"]
                t_training = results_i["comp_costs"]["t_training"]
                
                training_results_all.append({"model": model_type,
                                             "t_dimension": t_dimension,
                                             "t_instances": t_instances,
                                             "t_training": t_training, })
                
        
        training_results_all = pd.DataFrame(training_results_all)
    
    else:
        
        for results_file in files_list:
            
            with open(results_file, "rb") as f:
                
                results_i = pickle.load(f)
                
            model_type =  results_i["model_type"]
            t_dimension = results_i["t_dimension"]
            t_instances = results_i["t_instances"]
            t_training = results_i["comp_costs"]["t_training"]
            
            training_results_all.append({"model": model_type,
                                         "t_dimension": t_dimension,
                                         "t_instances": t_instances,
                                         "t_training": t_training})
            
    
    training_results_all = pd.DataFrame(training_results_all)
    
    return training_results_all
            


def unify_and_get_all_FE_costs(results_path, files_list = None):
    
    training_results_all = []
    
    if files_list == None:
        
        for results_file in os.listdir(results_path):
        
            if results_file.endswith(".pkl"):  
                
                results_file_path = os.path.join(results_path, results_file)
            
                with open(results_file_path, "rb") as f:
                    
                    results_i = pickle.load(f)
                    
                model_type =  results_i["model_type"]
                t_dimension = results_i["t_dimension"]
                t_instances = results_i["t_instances"]
                
                if model_type == "XGB":
                    t_FE = results_i["comp_costs"]["t_FE"] + results_i["comp_costs"]["t_FS"]
                else:
                    t_FE = 0

                
                training_results_all.append({"model": model_type,
                                             "t_dimension": t_dimension,
                                             "t_instances": t_instances,
                                             "t_FE": t_FE, })
                
        
        training_results_all = pd.DataFrame(training_results_all)
    
    else:
        
        for results_file in files_list:
            
            with open(results_file, "rb") as f:
                
                results_i = pickle.load(f)
                
            model_type =  results_i["model_type"]
            t_dimension = results_i["t_dimension"]
            t_instances = results_i["t_instances"]
            
            if model_type == "XGB":
                t_FE = results_i["comp_costs"]["t_FE"] + results_i["comp_costs"]["t_FS"]
            else:
                t_FE = 0

            
            training_results_all.append({"model": model_type,
                                         "t_dimension": t_dimension,
                                         "t_instances": t_instances,
                                         "t_FE": t_FE, })
            
    
    training_results_all = pd.DataFrame(training_results_all)
    
    return training_results_all
    
def unify_and_get_all_metrics(results_path, files_list = None):    
    
    training_results_all = []
    
    if files_list == None:
        
        for results_file in os.listdir(results_path):
        
            if results_file.endswith(".pkl"):  
                
                results_file_path = os.path.join(results_path, results_file)
            
                with open(results_file_path, "rb") as f:
                    
                    results_i = pickle.load(f)
                    
                model_type =  results_i["model_type"]
                t_dimension = results_i["t_dimension"]
                t_instances = results_i["t_instances"]
                r2_validation = results_i['model_metrics']['cv_metric'][1]
                r2_all = results_i['model_metrics']['global_metric'][1]
                
                training_results_all.append({"model": model_type,
                                             "t_dimension": t_dimension,
                                             "t_instances": t_instances,
                                             "r2_validation": r2_validation,
                                             "r2_all": r2_all})
                
        
        training_results_all = pd.DataFrame(training_results_all)
    
    else:
        
        for results_file in files_list:
            
            with open(results_file, "rb") as f:
                
                results_i = pickle.load(f)
                
            model_type =  results_i["model_type"]
            t_dimension = results_i["t_dimension"]
            t_instances = results_i["t_instances"]
            r2_validation = results_i['model_metrics']['cv_metric'][1]
            r2_all = results_i['model_metrics']['global_metric'][1]
            
            training_results_all.append({"model": model_type,
                                         "t_dimension": t_dimension,
                                         "t_instances": t_instances,
                                         "r2_validation": r2_validation,
                                         "r2_all": r2_all})
            
    
    training_results_all = pd.DataFrame(training_results_all)
    
    return training_results_all
    

def plot_comp_costs(training_results_all, model, save = False, saving_path = None):
    
    model_results = training_results_all[training_results_all["model"] == model]

    model_results = model_results.sort_values(by="t_dimension")

    
    plt.figure(figsize=(8, 5))
    plt.plot(model_results["t_dimension"], model_results["t_training"]/60, marker='o', linestyle='-', label=f"{model}")

    plt.xlabel("t_dimension")
    plt.ylabel("t_training (min)")
    plt.title(f"Training computational time costs for {model}")
    plt.legend()
    plt.grid(True)


    if save:
        
        os.makedirs(saving_path, exist_ok=True)
        plt.savefig(saving_path)
        plt.close()  
    
def plot_FE_and_training_costs(training_results_all, save = False, saving_path = None):

    training_results_all["label"] = training_results_all.apply(lambda row: f"{row['t_dimension']}\n{row['t_instances']}", axis=1)

    training_results_all = training_results_all.sort_values(by=["t_dimension", "t_instances"])

    x = np.arange(len(training_results_all))  
    width = 0.4  

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width/2, training_results_all["t_training"]/60, width, label="t_training", color="blue")
    ax.bar(x + width/2, training_results_all["t_FE"]/60, width, label="t_FE", color="orange")

    ax.set_xticks(x)
    ax.set_xticklabels(training_results_all["label"], rotation=45, ha="right")

    ax.set_xlabel("t_dimension - t_instances")
    ax.set_ylabel("Time (min)")
    ax.set_title("Computational costs for feature extraction and training")
    ax.legend()

    plt.tight_layout()
    plt.show()
    
    if save:
        
        os.makedirs(saving_path, exist_ok=True)
        plt.savefig(saving_path)
        plt.close()
           
def plot_r2_n_instances(training_results_all, model, save = False, saving_path = None):

    model_results = training_results_all[training_results_all["model"] == model]

    model_results = model_results.sort_values(by="t_dimension")

    
    plt.figure(figsize=(8, 5))
    plt.plot(model_results["t_dimension"], model_results["r2_validation"], marker='o', linestyle='-', label=f"{model}")

    plt.xlabel("t_dimension")
    plt.ylabel("R2 on validation")
    plt.title(f"R2 validation mean value {model}")
    plt.legend()
    plt.grid(True)


    if save:
        
        os.makedirs(saving_path, exist_ok=True)
        plt.savefig(saving_path)
        plt.close()  
        
    plt.figure(figsize=(8, 5))
    plt.plot(model_results["t_dimension"], model_results["r2_all"], marker='o', linestyle='-', label=f"{model}")

    plt.xlabel("t_dimension")
    plt.ylabel("R2 training all")
    plt.title(f"R2 training set values {model}")
    plt.legend()
    plt.grid(True)
    
    if save:
        
        os.makedirs(saving_path, exist_ok=True)
        plt.savefig(saving_path)
        plt.close()

