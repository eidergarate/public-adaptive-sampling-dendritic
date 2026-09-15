# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:54:28 2024

@author: egarate
"""

import pandas as pd
import numpy as np
import time

def feature_selection(data_tabular, FS_method):
    
    if FS_method == "XGB_importances":
        
        data_filtered = data_tabular
        del data_tabular
        t_fs = 0
        
    return data_filtered, t_fs