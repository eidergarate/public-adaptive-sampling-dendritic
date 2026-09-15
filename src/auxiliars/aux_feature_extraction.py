# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:44:03 2024

@author: egarate
"""

import pandas as pd
import numpy as np
from src.auxiliars.aux_constructDF_from_phase_field import *
import time


def feature_extraction(neighs_t, data_raw_path, simulations, FE_method):
    
    if FE_method == "radius_method":
        
        t_ini = time.time()
        data_FE = constructSpatioTemporal_all(r=2, neighs_t=neighs_t, parametrosSimulaciones=simulations, data_path=data_raw_path, data_raw = data_raw_path)
        t_f = time.time() - t_ini
        
    return data_FE, t_f
        
        
        