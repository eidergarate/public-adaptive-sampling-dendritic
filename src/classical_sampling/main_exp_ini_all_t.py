# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:34:51 2024

@author: egarate
"""

import os
import sys

wdir = "/home/azureuser/repos/Tesis_adaptive_sampling"

os.chdir(wdir)

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.classical_sampling.main_classical_all_t_instances import *
from src.green_computing.codecarbon_emission_tracking import EmissionTrackerManager


import warnings
warnings.filterwarnings("ignore")


# XGB TABULAR
FE_method="radius_method" 
FS_method="XGB_importances"
model="XGB"
symmetry={'aniso4' : 4, 'aniso6' : 6}
loss_function="reg:squarederror"
save_dataset=True
model_metrics={'cv_metric' : ["r2"], "global_metric" : ["r2_score"]}
saving_paths = {'doe': wdir + '/data/sampling',
                'model_path': wdir + '/outputs/classic/models',
                'results_saving_path': wdir + '/outputs/classic/results',
                'data_raw': wdir + '/data/raw/classic'}   
model_hyperparameters = None

n_points = [70, 150]
EmissionTrackerManager(saving_paths['results_saving_path'])

for n_point_exp in n_points:
    
    main_classical_sampling_and_modeling_all_t_instances(n_point_exp, FE_method, FS_method,
                                              model, symmetry, loss_function, save_dataset, 
                                              model_metrics, model_hyperparameters,
                                              saving_paths
                                              )
    
    
    
    
#CNN
FE_method="radius_method" 
FS_method="XGB_importances"
model="CNN_base"
symmetry={'aniso4' : 4, 'aniso6' : 6}
loss_function="mse"
save_dataset=True
model_metrics={'cv_metric' : ["r2"], "global_metric" : ["r2_score"]}


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



n_points = [70, 150, 300, 500, 700, 1000, 1300, 1600, 2000]


for n_point_exp in n_points:
    
    
    model_hyperparameters = {'epochs' : 100,
                             'batch_size' : n_point_exp//2,
                             'validation_split' : 0.7,
                             'CNN' : CNN_hyperparameters,
                             'tabular' : tabular_hyperparameters,
                             'optimizer': 'adam',
                             "loss_function" : "mse",
        
        }
    
    main_classical_sampling_and_modeling_all_t_instances(n_point_exp, FE_method, FS_method,
                                             model, symmetry, loss_function, save_dataset, 
                                             model_metrics, model_hyperparameters,
                                             saving_paths
                                             )
    
    
    
    
#CNN SELF SUPERVISED
FE_method="radius_method" 
FS_method="XGB_importances"
model="CNN_self_supervised"
symmetry={'aniso4' : 4, 'aniso6' : 6}
loss_function="mse"
save_dataset=True
model_metrics={'cv_metric' : ["r2"], "global_metric" : ["r2_score"]}


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

  
  
n_points = [1300, 1600, 2000]


for n_point_exp in n_points:
    
    model_hyperparameters = {'epochs' : 100,
                                'batch_size' : n_point_exp//2,
                                'validation_split' : 0.7,
                                'CNN' : CNN_hyperparameters,
                                'tabular' : tabular_hyperparameters,                                 'optimizer': 'adam',
                                "loss_function" : "mse",
          
          }  
      
    main_classical_sampling_and_modeling_all_t_instances(n_point_exp, FE_method, FS_method,
                                                model, symmetry, loss_function, save_dataset, 
                                                model_metrics, model_hyperparameters,
                                                saving_paths
                                                )