# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 08:47:07 2025

@author: egarate
"""

import os
import sys

wdir = "/home/azureuser/repos/Tesis_adaptive_sampling"

os.chdir(wdir)

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.adaptive_sampling.main_adaptive_sampling_all_t_instances import *
from src.green_computing.codecarbon_emission_tracking import EmissionTrackerManager


n_ini_points = 70
p_adding_points = 0.15
n_samples_max = 700
t_instances = [200, 1000, 2000, 4000]
FE_method="radius_method"
model = "CNN_MC_dropout"
FE_method="radius_method"
FS_method=None
symmetry={'aniso4' : 4, 'aniso6' : 6}
loss_function="mse"
save_dataset=True
model_metrics={'cv_metric' : [], "global_metric" : []}


CNN_hyperparameter_image_i = {'filters' : 32,
                              'kernel_size' : (3,3),
                              'strides' : (1,1),
                              'activation' : 'relu'}

CNN_hyperparameters_combined = {'filters' : 64,
                                'kernel_size' : (3,3),
                                'strides' : (1,1),
                                'activation' : 'relu'}

CNN_hyperparameters = [CNN_hyperparameter_image_i, CNN_hyperparameters_combined]

tabular_hyperparameters = {'neurons' : [32, 16],
                           'activation' : ['relu', 'relu']}

model_hyperparameters = {'epochs' : 100,
                         'batch_size' : 30,
                         'validation_split' : 0.7,
                         'CNN' : CNN_hyperparameters,
                         'tabular' : tabular_hyperparameters,
                         'optimizer': 'adam',
                         "loss_function" : "mse"}
incremental = True
n_retrain_all = 50
retrain_some = False
saving_paths = {'doe': wdir + '/data',
                'model_path': wdir + '/outputs/adaptive/models',
                'results_saving_path': wdir + '/outputs/adaptive/results',
                'data_raw': wdir + '/data/raw/adaptive'}

EmissionTrackerManager(saving_paths['results_saving_path'])
main_adaptive_sampling_and_modeling_all_t_instances(n_ini_points, p_adding_points, n_samples_max, 
                                        FE_method, FS_method, model, symmetry, loss_function, save_dataset, 
                                         model_metrics, model_hyperparameters, saving_paths,
                                         incremental, n_retrain_all, retrain_some)
 

#XGBoost

n_ini_points = 70
p_adding_points = 0.15
n_samples_max = 700
t_instances = [200, 1000, 2000, 4000]
FE_method="radius_method"
model = "XGB_probabilistic"
FE_method="radius_method"
FS_method="XGB_importances"
symmetry={'aniso4' : 4, 'aniso6' : 6}
loss_function="reg:squarederror"
save_dataset=True
model_metrics={'cv_metric' : [], "global_metric" : []}


model_hyperparameters = None

incremental = True
n_retrain_all = 50
retrain_some = False
saving_paths = {'doe': wdir + '/data',
                'model_path': wdir + '/outputs/adaptive/models',
                'results_saving_path': wdir + '/outputs/adaptive/results',
                'data_raw': wdir + '/data/raw/adaptive'}


main_adaptive_sampling_and_modeling_all_t_instances(n_ini_points, p_adding_points, n_samples_max, 
                                       FE_method, FS_method, model, symmetry, loss_function, save_dataset, 
                                        model_metrics, model_hyperparameters, saving_paths,
                                        incremental, n_retrain_all, retrain_some)


# SELF SUPERVISED CNN
model = "CNN_MC_dropout_ss"
n_ini_points = 70
p_adding_points = 0.15
n_samples_max = 700
t_instances = [200, 1000, 2000, 4000]
FE_method="radius_method"
FE_method="radius_method"
FS_method=None
symmetry={'aniso4' : 4, 'aniso6' : 6}
loss_function="mse"
save_dataset=True
model_metrics={'cv_metric' : [], "global_metric" : []}


CNN_hyperparameter_image_i = {'filters' : 32,
                              'kernel_size' : (3,3),
                              'strides' : (1,1),
                              'activation' : 'relu'}

CNN_hyperparameters_combined = {'filters' : 64,
                                'kernel_size' : (3,3),
                                'strides' : (1,1),
                                'activation' : 'relu'}

CNN_hyperparameters = [CNN_hyperparameter_image_i, CNN_hyperparameters_combined]

tabular_hyperparameters = {'neurons' : [32, 16],
                           'activation' : ['relu', 'relu']}

model_hyperparameters = {'epochs' : 100,
                         'batch_size' : 30,
                         'validation_split' : 0.7,
                         'CNN' : CNN_hyperparameters,
                         'tabular' : tabular_hyperparameters,
                         'optimizer': 'adam',
                         "loss_function" : "mse"}
incremental = True
n_retrain_all = 50
retrain_some = False
saving_paths = {'doe': wdir + '/data',
                'model_path': wdir + '/outputs/adaptive/models',
                'results_saving_path': wdir + '/outputs/adaptive/results',
                'data_raw': wdir + '/data/raw/adaptive'}


main_adaptive_sampling_and_modeling_all_t_instances(n_ini_points, p_adding_points, n_samples_max, 
                                        FE_method, FS_method, model, symmetry, loss_function, save_dataset, 
                                         model_metrics, model_hyperparameters, saving_paths,
                                         incremental, n_retrain_all, retrain_some)










