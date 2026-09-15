# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:54:54 2024

@author: egarate
"""

from physical_model.runMatlab import simulateExp
#from oct2py import Oct2Py
import pandas as pd
import numpy as np


def simulate_phase_field_experiments(doe_path, data_raw_path, idxToSave):    
    idxToSave_octave = list(idxToSave)
    
    experiments = pd.read_csv(doe_path)
    
    Nx = 100
    Ny = 100
    writeVTK = False
    nargout = 3
    gamma = 10.0
    
    for experiment_id in range(experiments.shape[0]):
        tau_i = experiments.loc[experiment_id, 'tau']
        epsilon_i = experiments.loc[experiment_id, 'epsilonbar']
        kappa_i = experiments.loc[experiment_id, 'kappa']
        delta_i = experiments.loc[experiment_id, 'delta']
        theta0_i = experiments.loc[experiment_id, 'theta0']
        alpha_i = experiments.loc[experiment_id, 'alpha']
        j_aniso_i = experiments.loc[experiment_id, 'j_aniso']
        
        simulateExp(None, tau_i, epsilon_i, kappa_i, delta_i, j_aniso_i, alpha_i,
                       theta0_i, experiment_id, idxToSave_octave, gamma,
                       writeVTK, nargout, Nx, Ny, data_raw_path)
    
    return True

def simulate_phase_field_adaptive(new_added_points, doe_path, data_raw_path, idxToSave):
    
    idxToSave_octave = list(idxToSave)
    
    experiments = new_added_points
    
    Nx = 100
    Ny = 100
    writeVTK = False
    nargout = 3
    gamma = 10.0
    
    for index, row in experiments.iterrows():
        
        experiment_id = experiments.loc[index, 'exp_id']
        tau_i = experiments.loc[index, 'tau']
        epsilon_i = experiments.loc[index, 'epsilonbar']
        kappa_i = experiments.loc[index, 'kappa']
        delta_i = experiments.loc[index, 'delta']
        theta0_i = experiments.loc[index, 'theta0']
        alpha_i = experiments.loc[index, 'alpha']
        j_aniso_i = experiments.loc[index, 'j_aniso']
        
        simulateExp(None, tau_i, epsilon_i, kappa_i, delta_i, j_aniso_i, alpha_i,
                       theta0_i, experiment_id, idxToSave_octave, gamma,
                       writeVTK, nargout, Nx, Ny, data_raw_path)
    
    return True
    
    
    
    