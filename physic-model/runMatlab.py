# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 09:39:01 2022

@author: Jon_l, egarate
"""

import os
import numpy as np
import subprocess

#Function to simulate an specific experiment by the Phase Field Model implemented in MATLAB

#Inputs: 
#   PHYSICAL MODEL'S PARAMETERS

#   tau (time constant), epsilonb (epsilon's mean value), kappa (dimensionless latent heat),
#   delta (anisotropy strenght), aniso (cubic or hexagonal), alpha, theta0 (initial missorientation) and gamma
#   gamma's default value is set to 10 based on the literature
#   aniso € {4, 6}; alpha € [0, 1]
#   Use article search space.

#   testNumber := is the identification number for the simulated experiment
#   saveSteps := The bumber of instances on time that are going to be saved. Default is 2 and partial results in t = 100, 4000
#                If saveSteps > 2, uniformly placed saveSteps instances between [100, 4000] are going to be saved
#   writeVTK := A boolean variable describing if some simulation logs are needed to save. Default is False.
#   nargout := (?)
#   Nx := number of grids defined in x (Default is 300).
#   Ny := Number of gids defined in y (Default is 300).


#Outputs: Final temperatures and OP's for the material + nedded computation time

def simulateExp(oc, tau, epsilonb, kappa, delta, aniso, alpha,  theta0, testNumber, idxToSave, gamma = 10.0, writeVTK = False, nargout = 3,  Nx = 300, Ny = 300, data_raw_path = None):
    
    #Define solution domain and discretization parameters

    idxToSave_octave = f"[{','.join(map(str, idxToSave))}]"
    writeVTK_octave = "false" if not writeVTK else "true"
    
    # path to `dendriticSolidification.m`
    physical_model_path = "./physical_model" 

    if os.name == 'nt': # nt is Windows
    
        octave_command = "C:/Octave-6.4.0/mingw64/bin/octave-cli.exe" # Change to your octave directory
        
        if not os.path.exists(octave_command):
            raise FileNotFoundError(f"Octave wasn't found in Windows: {octave_command}")
    else:
        octave_command = r"/usr/bin/octave-cli" #This is for a linux system, change if needed
        if not os.path.exists(octave_command):
            raise FileNotFoundError(f"Octave wasn't found in Linux: {octave_command}")

    # Octave code needs to be a string
    
    command = [
        octave_command, "--eval",
        f"addpath('{physical_model_path}'); "
        f"dendriticSolidification({testNumber}, {tau}, {epsilonb}, {kappa}, {delta}, {aniso}, "
        f"{alpha}, {gamma}, {theta0}, {Nx}, {Ny}, 0.03, 0.03, {writeVTK_octave}, "
        f"{idxToSave_octave}, '{data_raw_path}');"
    ]
    

    # Call to octave and run
    try:
        subprocess.run(command, check=True)
        print(f"Simulación {testNumber} completada.")
    except subprocess.CalledProcessError as e:
        print(f"Error en la simulación {testNumber}. Revisa 'octave_log.txt' para más detalles.")
        raise


