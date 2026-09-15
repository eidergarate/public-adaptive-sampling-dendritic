# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
        

## Function to input each data point
# INPUTS: xi: a dataframe row from data/df_1000_clean_renamed.csv
#         n_grid: number of nxn grid of the image. Default is 100.
# OUTPUTS: xi imputed if is a image edge pixel value

def imput_data_point(xi, t_instances = [1000, 2000, 4000]):
    
    t_instances = list(map(str, t_instances))
    
    if xi['x'] == 1:
        
        if xi['y'] == 1:
            
            for t in t_instances:
                
                for var in ["OP", "T"]:
                
                    xi[var + '_u1_' + t] = xi[var + '_' + t]
                    xi[ var + '_u2_' + t] = xi[var + '_' + t]
                    xi[var + '_u1r1_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_u2r1_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_u1r2_' + t] = xi[var + '_r2_' + t]
                    xi[var + '_u2r2_' + t] = xi[var + '_r2_' + t]
                    
                    xi[var + '_l1_' + t] = xi[var + '_' + t]
                    xi[var + '_l2_' + t] = xi[var + '_' + t]
                    xi[var + '_d1l1_' + t] = xi[var + '_d1_' + t]
                    xi[var + '_d1l2_' + t] = xi[var + '_d1_' + t]
                    xi[var + '_d2l1_' + t] = xi[var + '_d2_' + t]
                    xi[var + '_d2l2_' + t] = xi[var + '_d2_' + t]
                    
                    xi[var + "_u1l1_" + t] = xi[var + '_' + t]
                    xi[var + "_u1l2_" + t] = xi[var + '_' + t]
                    xi[var + "_u2l1_" + t] = xi[var + '_' + t]
                    xi[var + "_u2l2_" + t] = xi[var + '_' + t]
                    
        elif xi['y'] == 2:
            
            for t in t_instances:
                
                for var in ["OP", "T"]:
                
                    xi[var + '_u1_' + t] = xi[var + '_' + t]
                    xi[ var + '_u2_' + t] = xi[var + '_' + t]
                    xi[var + '_u1r1_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_u2r1_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_u1r2_' + t] = xi[var + '_r2_' + t]
                    xi[var + '_u2r2_' + t] = xi[var + '_r2_' + t]
                    xi[var + "_u1l1_" + t] = xi[var + '_l1_' + t]
                    xi[var + "_u2l1_" + t] = xi[var + '_l1_' + t]
                        
                    xi[var + '_l2_' + t] = xi[var + '_l1_' + t]
                    xi[var + '_d1l2_' + t] = xi[var + '_d1l1_' + t]
                    xi[var + '_d2l2_' + t] = xi[var + '_d2l1_' + t]
                    
                    
                    xi[var + "_u1l2_" + t] = xi[var + '_l1_' + t]
                    xi[var + "_u2l2_" + t] = xi[var + '_l1_' + t]
                    
        elif (xi['y'] > 2) & (xi['y'] < 99):    
            
            for t in t_instances:
                
                for var in ["OP", "T"]:
                    
                    xi[var + "_u1l1_" + t] = xi[var + '_l1_' + t]
                    xi[var + "_u2l1_" + t] = xi[var + '_l1_' + t]
                    xi[var + "_u1l2_" + t] = xi[var + '_l2_' + t]
                    xi[var + "_u2l2_" + t] = xi[var + '_l2_' + t]
                    xi[var + '_u1_' + t] = xi[var + '_' + t]
                    xi[ var + '_u2_' + t] = xi[var + '_' + t]
                    xi[var + '_u1r1_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_u2r1_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_u1r2_' + t] = xi[var + '_r2_' + t]
                    xi[var + '_u2r2_' + t] = xi[var + '_r2_' + t]
        
        elif xi['y'] == 99:
            
            for t in t_instances:
                
                for var in ["OP", "T"]:
                    
                    xi[var + "_u1l1_" + t] = xi[var + '_l1_' + t]
                    xi[var + "_u2l1_" + t] = xi[var + '_l1_' + t]
                    xi[var + "_u1l2_" + t] = xi[var + '_l2_' + t]
                    xi[var + "_u2l2_" + t] = xi[var + '_l2_' + t]
                    xi[var + '_u1_' + t] = xi[var + '_' + t]
                    xi[ var + '_u2_' + t] = xi[var + '_' + t]
                    xi[var + '_u1r1_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_u2r1_' + t] = xi[var + '_r1_' + t]
                    
                    xi[ var + '_r2_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_d1r2_' + t] = xi[var + '_d1r1_' + t]
                    xi[var + '_d2r2_' + t] = xi[var + '_d2r1_' + t]
                    
                    xi[var + '_u1r2_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_u2r2_' + t] = xi[var + '_r1_' + t]
           
        elif xi['y'] == 100:
                       
            for t in t_instances:
                           
                for var in ["OP", "T"]:
                               
                    xi[var + '_u1_' + t] = xi[var + '_' + t]
                    xi[ var + '_u2_' + t] = xi[var + '_' + t]
                    xi[var + '_u1l1_' + t] = xi[var + '_l1_' + t]
                    xi[var + '_u2l1_' + t] = xi[var + '_l1_' + t]
                    xi[var + '_u1l2_' + t] = xi[var + '_l2_' + t]
                    xi[var + '_u2l2_' + t] = xi[var + '_l2_' + t]
                    
                    xi[var + '_r1_' + t] = xi[var + '_' + t]
                    xi[var + '_r2_' + t] = xi[var + '_' + t]
                    xi[var + '_d1r1_' + t] = xi[var + '_d1_' + t]
                    xi[var + '_d1r2_' + t] = xi[var + '_d1_' + t]
                    xi[var + '_d2r1_' + t] = xi[var + '_d2_' + t]
                    xi[var + '_d2r2_' + t] = xi[var + '_d2_' + t]
                    
                    xi[var + "_u1r1_" + t] = xi[var + '_' + t]
                    xi[var + "_u1r2_" + t] = xi[var + '_' + t]
                    xi[var + "_u2r1_" + t] = xi[var + '_' + t]
                    xi[var + "_u2r2_" + t] = xi[var + '_' + t]
                    
    elif xi['x'] == 2:
        
        if xi['y'] == 1:  
            
            for t in t_instances:
                
                for var in ["OP", "T"]:
                    
                    xi[var + "_u1l1_" + t] = xi[var + '_u1_' + t]
                    xi[var + "_u1l2_" + t] = xi[var + '_u1_' + t]
                    xi[var + "_l1_" + t] = xi[var + '_' + t]
                    xi[var + "_l2_" + t] = xi[var + '_' + t]
                    xi[var + "_d1l1_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_d1l2_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_d2l1_" + t] = xi[var + '_d2_' + t]
                    xi[var + "_d2l2_" + t] = xi[var + '_d2_' + t]
                    
                    xi[var + "_u2_" + t] = xi[var + '_u1_' + t]
                    xi[var + "_u2r1_" + t] = xi[var + '_u1r1_' + t]
                    xi[var + "_u2r2_" + t] = xi[var + '_u1r2_' + t]
                    
                    xi[var + "_u2l1_" + t] = xi[var + '_u1_' + t]
                    xi[var + "_u2l2_" + t] = xi[var + '_u1_' + t]
                    
        if xi['y'] == 2:         
                    
            for t in t_instances:
                        
                for var in ["OP", "T"]:
                    
                    xi[var + "_u2l1_" + t] = xi[var + '_u1l1_' + t]
                    xi[var + "_u2_" + t] = xi[var + '_u1_' + t]
                    xi[var + "_u2r1_" + t] = xi[var + '_u1r1_' + t]
                    xi[var + "_u2r2_" + t] = xi[var + '_u1r2_' + t]
                    
                    xi[var + "_u1l2_" + t] = xi[var + '_u1l1_' + t]
                    xi[var + "_l2_" + t] = xi[var + '_l1_' + t]
                    xi[var + "_d1l2_" + t] = xi[var + '_d1l1_' + t]
                    xi[var + "_d2l2_" + t] = xi[var + '_d2l1_' + t]
                            
                    xi[var + "_u2l2_" + t] = xi[var + '_u1l1_' + t]
                    
        if (xi['y'] > 2) & (xi['y'] < 99):         
                    
            for t in t_instances:
                        
                for var in ["OP", "T"]:
                    
                    xi[var + "_u2l2_" + t] = xi[var + '_u1l2_' + t]
                    xi[var + "_u2l1_" + t] = xi[var + '_u1l1_' + t]
                    xi[var + "_u2_" + t] = xi[var + '_u1_' + t]
                    xi[var + "_u2r1_" + t] = xi[var + '_u1r1_' + t]
                    xi[var + "_u2r2_" + t] = xi[var + '_u1r2_' + t]
                            
        if xi['y'] == 99:         
                    
            for t in t_instances:
                        
                for var in ["OP", "T"]:
                            
                    xi[var + "_u2l2_" + t] = xi[var + '_u1l2_' + t]
                    xi[var + "_u2l1_" + t] = xi[var + '_u1l1_' + t]
                    xi[var + "_u2_" + t] = xi[var + '_u1_' + t]
                    xi[var + "_u2r1_" + t] = xi[var + '_u1r1_' + t]
                     
                    xi[var + "_u1r2_" + t] = xi[var + '_u1r1_' + t]
                    xi[var + "_r2_" + t] = xi[var + '_r1_' + t]
                    xi[var + "_d1r2_" + t] = xi[var + '_d1r1_' + t]
                    xi[var + "_d2r2_" + t] = xi[var + '_d2r1_' + t]
                             
                    xi[var + "_u2r2_" + t] = xi[var + '_u1r1_' + t]       
            
        if xi['y'] == 100:         
                    
            for t in t_instances:
                        
                for var in ["OP", "T"]:
                            
                          xi[var + "_u1r1_" + t] = xi[var + '_u1_' + t]
                          xi[var + "_u1r2_" + t] = xi[var + '_u1_' + t]
                          xi[var + "_r1_" + t] = xi[var + '_' + t]
                          xi[var + "_r2_" + t] = xi[var + '_' + t]
                          xi[var + "_d1r1_" + t] = xi[var + '_d1_' + t]
                          xi[var + "_d1r2_" + t] = xi[var + '_d1_' + t]
                          xi[var + "_d2r1_" + t] = xi[var + '_d2_' + t]
                          xi[var + "_d2r2_" + t] = xi[var + '_d2_' + t]
                          
                          xi[var + "_u2_" + t] = xi[var + '_u1_' + t]
                          xi[var + "_u2l1_" + t] = xi[var + '_u1l1_' + t]
                          xi[var + "_u2l2_" + t] = xi[var + '_u1l2_' + t]
                          
                          xi[var + "_u2r1_" + t] = xi[var + '_u1_' + t]
                          xi[var + "_u2r2_" + t] = xi[var + '_u1_' + t]  
                            
    elif (xi['x'] > 2) & (xi['x'] < 99):
        
        if xi['y'] == 1:
            
            for t in t_instances:
                
                for var in ["OP", "T"]:
                    
                    xi[var + "_u2l2_" + t] = xi[var + '_u2_' + t]
                    xi[var + "_u1l2_" + t] = xi[var + '_u1_' + t]
                    xi[var + "_l2_" + t] = xi[var + '_' + t]
                    xi[var + "_d1l2_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_d2l2_" + t] = xi[var + '_d2_' + t]
                    
                    xi[var + "_u2l1_" + t] = xi[var + '_u2_' + t]
                    xi[var + "_u1l1_" + t] = xi[var + '_u1_' + t]
                    xi[var + "_l1_" + t] = xi[var + '_' + t]
                    xi[var + "_d1l1_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_d2l1_" + t] = xi[var + '_d2_' + t]
                                       
        if xi['y'] == 2:
            
            for t in t_instances:
                
                for var in ["OP", "T"]:
                    
                    xi[var + "_u2l2_" + t] = xi[var + '_u2l1_' + t]
                    xi[var + "_u1l2_" + t] = xi[var + '_u1l1_' + t]
                    xi[var + "_l2_" + t] = xi[var + '_l1_' + t]
                    xi[var + "_d1l2_" + t] = xi[var + '_d1l1_' + t]
                    xi[var + "_d2l2_" + t] = xi[var + '_d2l1_' + t]        
                            
        if xi['y'] == 99:
            
            for t in t_instances:
                
                for var in ["OP", "T"]:
            
                    xi[var + "_u2r2_" + t] = xi[var + '_u2r1_' + t]
                    xi[var + "_u1r2_" + t] = xi[var + '_u1r1_' + t]
                    xi[var + "_r2_" + t] = xi[var + '_' + t]
                    xi[var + "_d1r2_" + t] = xi[var + '_d1r1_' + t]
                    xi[var + "_d2r2_" + t] = xi[var + '_d2r1_' + t]
                    
        if xi['y'] == 100:
            
            for t in t_instances:
                
                for var in ["OP", "T"]:
            
                    xi[var + "_u2r2_" + t] = xi[var + '_u2_' + t]
                    xi[var + "_u1r2_" + t] = xi[var + '_u1_' + t]
                    xi[var + "_r2_" + t] = xi[var + '_' + t]
                    xi[var + "_d1r2_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_d2r2_" + t] = xi[var + '_d2_' + t]
                    
                    xi[var + "_u2r1_" + t] = xi[var + '_u2_' + t]
                    xi[var + "_u1r1_" + t] = xi[var + '_u1_' + t]
                    xi[var + "_r1_" + t] = xi[var + '_' + t]
                    xi[var + "_d1r1_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_d2r1_" + t] = xi[var + '_d2_' + t]
                    
    elif xi['x'] == 99:
        
        if xi['y'] == 1:  
            
            for t in t_instances:
                
                for var in ["OP", "T"]:
                    
                    xi[var + "_d1l1_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_d1l2_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_l1_" + t] = xi[var + '_' + t]
                    xi[var + "_l2_" + t] = xi[var + '_' + t]
                    xi[var + "_u1l1_" + t] = xi[var + '_u1_' + t]
                    xi[var + "_u1l2_" + t] = xi[var + '_u1_' + t]
                    xi[var + "_u2l1_" + t] = xi[var + '_u2_' + t]
                    xi[var + "_u2l2_" + t] = xi[var + '_u2_' + t]
                    
                    xi[var + "_d2_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_d2r1_" + t] = xi[var + '_d1r1_' + t]
                    xi[var + "_d2r2_" + t] = xi[var + '_d1r2_' + t]
                    
                    xi[var + "_d2l1_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_d2l2_" + t] = xi[var + '_d1_' + t]
                       
        if xi['y'] == 2:         
                    
            for t in t_instances:
                        
                for var in ["OP", "T"]:
                    
                    xi[var + "_d2l1_" + t] = xi[var + '_d1l1_' + t]
                    xi[var + "_d2_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_d2r1_" + t] = xi[var + '_d1r1_' + t]
                    xi[var + "_d2r2_" + t] = xi[var + '_d1r2_' + t]
                    
                    xi[var + "_d1l2_" + t] = xi[var + '_d1l1_' + t]
                    xi[var + "_l2_" + t] = xi[var + '_l1_' + t]
                    xi[var + "_u1l2_" + t] = xi[var + '_u1l1_' + t]
                    xi[var + "_u2l2_" + t] = xi[var + '_u2l1_' + t]
                            
                    xi[var + "_d2l2_" + t] = xi[var + '_d1l1_' + t]
                    
        if (xi['y'] > 2) & (xi['y'] < 99):         
                    
            for t in t_instances:
                        
                for var in ["OP", "T"]:
                    
                    xi[var + "_d2l2_" + t] = xi[var + '_d1l2_' + t]
                    xi[var + "_d2l1_" + t] = xi[var + '_d1l1_' + t]
                    xi[var + "_d2_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_d2r1_" + t] = xi[var + '_d1r1_' + t]
                    xi[var + "_d2r2_" + t] = xi[var + '_d1r2_' + t]
                            
        if xi['y'] == 99:         
                    
            for t in t_instances:
                        
                for var in ["OP", "T"]:
                            
                    xi[var + "_d2l2_" + t] = xi[var + '_d1l2_' + t]
                    xi[var + "_d2l1_" + t] = xi[var + '_d1l1_' + t]
                    xi[var + "_d2_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_d2r1_" + t] = xi[var + '_d1r1_' + t]
                     
                    xi[var + "_d1r2_" + t] = xi[var + '_d1r1_' + t]
                    xi[var + "_r2_" + t] = xi[var + '_r1_' + t]
                    xi[var + "_u1r2_" + t] = xi[var + '_u1r1_' + t]
                    xi[var + "_u2r2_" + t] = xi[var + '_u2r1_' + t]
                             
                    xi[var + "_d2r2_" + t] = xi[var + '_d1r1_' + t] 
                            
        if xi['y'] == 100:         
                     
            for t in t_instances:
                         
                for var in ["OP", "T"]:
                             
                    xi[var + "_d1r1_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_d1r2_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_r1_" + t] = xi[var + '_' + t]
                    xi[var + "_r2_" + t] = xi[var + '_' + t]
                    xi[var + "_u1r1_" + t] = xi[var + '_u1_' + t]
                    xi[var + "_u1r2_" + t] = xi[var + '_u1_' + t]
                    xi[var + "_u2r1_" + t] = xi[var + '_u2_' + t]
                    xi[var + "_u2r2_" + t] = xi[var + '_u2_' + t]
                           
                    xi[var + "_d2_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_d2l1_" + t] = xi[var + '_d1l1_' + t]
                    xi[var + "_d2l2_" + t] = xi[var + '_d1l2_' + t]
                           
                    xi[var + "_d2r1_" + t] = xi[var + '_d1_' + t]
                    xi[var + "_d2r2_" + t] = xi[var + '_d1_' + t]                    
                            
    elif xi['x'] == 100:
        
        if xi['y'] == 1:
            
            for t in t_instances:
                
                for var in ["OP", "T"]:
                
                    xi[var + '_d1_' + t] = xi[var + '_' + t]
                    xi[ var + '_d2_' + t] = xi[var + '_' + t]
                    xi[var + '_d1r1_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_d2r1_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_d1r2_' + t] = xi[var + '_r2_' + t]
                    xi[var + '_d2r2_' + t] = xi[var + '_r2_' + t]
                    
                    xi[var + '_l1_' + t] = xi[var + '_' + t]
                    xi[var + '_l2_' + t] = xi[var + '_' + t]
                    xi[var + '_u1l1_' + t] = xi[var + '_u1_' + t]
                    xi[var + '_u1l2_' + t] = xi[var + '_u1_' + t]
                    xi[var + '_u2l1_' + t] = xi[var + '_u2_' + t]
                    xi[var + '_u2l2_' + t] = xi[var + '_u2_' + t]
                    
                    xi[var + "_d1l1_" + t] = xi[var + '_' + t]
                    xi[var + "_d1l2_" + t] = xi[var + '_' + t]
                    xi[var + "_d2l1_" + t] = xi[var + '_' + t]
                    xi[var + "_d2l2_" + t] = xi[var + '_' + t]
                    
        elif xi['y'] == 2:
            
            for t in t_instances:
                
                for var in ["OP", "T"]:
                
                    xi[var + '_d1_' + t] = xi[var + '_' + t]
                    xi[ var + '_d2_' + t] = xi[var + '_' + t]
                    xi[var + '_d1r1_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_d2r1_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_d1r2_' + t] = xi[var + '_r2_' + t]
                    xi[var + '_d2r2_' + t] = xi[var + '_r2_' + t]
                    xi[var + "_d1l1_" + t] = xi[var + '_l1_' + t]
                    xi[var + "_d2l1_" + t] = xi[var + '_l1_' + t]
                        
                    xi[var + '_l2_' + t] = xi[var + '_l1_' + t]
                    xi[var + '_u1l2_' + t] = xi[var + '_u1l1_' + t]
                    xi[var + '_u2l2_' + t] = xi[var + '_u2l1_' + t]
                    
                    
                    xi[var + "_d1l2_" + t] = xi[var + '_l1_' + t]
                    xi[var + "_d2l2_" + t] = xi[var + '_l1_' + t]
                    
        elif (xi['y'] > 2) & (xi['y'] < 99):    
            
            for t in t_instances:
                
                for var in ["OP", "T"]:
                    
                    xi[var + "_d1l1_" + t] = xi[var + '_l1_' + t]
                    xi[var + "_d2l1_" + t] = xi[var + '_l1_' + t]
                    xi[var + "_d1l2_" + t] = xi[var + '_l2_' + t]
                    xi[var + "_d2l2_" + t] = xi[var + '_l2_' + t]
                    xi[var + '_d1_' + t] = xi[var + '_' + t]
                    xi[ var + '_d2_' + t] = xi[var + '_' + t]
                    xi[var + '_d1r1_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_d2r1_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_d1r2_' + t] = xi[var + '_r2_' + t]
                    xi[var + '_d2r2_' + t] = xi[var + '_r2_' + t]
        
        elif xi['y'] == 99:
            
            for t in t_instances:
                
                for var in ["OP", "T"]:
                    
                    xi[var + "_d1l1_" + t] = xi[var + '_l1_' + t]
                    xi[var + "_d2l1_" + t] = xi[var + '_l1_' + t]
                    xi[var + "_d1l2_" + t] = xi[var + '_l2_' + t]
                    xi[var + "_d2l2_" + t] = xi[var + '_l2_' + t]
                    xi[var + '_d1_' + t] = xi[var + '_' + t]
                    xi[ var + '_d2_' + t] = xi[var + '_' + t]
                    xi[var + '_d1r1_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_d2r1_' + t] = xi[var + '_r1_' + t]
                    
                    xi[ var + '_r2_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_u1r2_' + t] = xi[var + '_u1r1_' + t]
                    xi[var + '_u2r2_' + t] = xi[var + '_u2r1_' + t]
                    
                    xi[var + '_d1r2_' + t] = xi[var + '_r1_' + t]
                    xi[var + '_d2r2_' + t] = xi[var + '_r1_' + t]
           
        elif xi['y'] == 100:
                       
            for t in t_instances:
                           
                for var in ["OP", "T"]:
                               
                    xi[var + '_d1_' + t] = xi[var + '_' + t]
                    xi[ var + '_d2_' + t] = xi[var + '_' + t]
                    xi[var + '_d1l1_' + t] = xi[var + '_l1_' + t]
                    xi[var + '_d2l1_' + t] = xi[var + '_l1_' + t]
                    xi[var + '_d1l2_' + t] = xi[var + '_l2_' + t]
                    xi[var + '_d2l2_' + t] = xi[var + '_l2_' + t]
                    
                    xi[var + '_r1_' + t] = xi[var + '_' + t]
                    xi[var + '_r2_' + t] = xi[var + '_' + t]
                    xi[var + '_u1r1_' + t] = xi[var + '_u1_' + t]
                    xi[var + '_u1r2_' + t] = xi[var + '_u1_' + t]
                    xi[var + '_u2r1_' + t] = xi[var + '_u2_' + t]
                    xi[var + '_u2r2_' + t] = xi[var + '_u2_' + t]
                    
                    xi[var + "_d1r1_" + t] = xi[var + '_' + t]
                    xi[var + "_d1r2_" + t] = xi[var + '_' + t]
                    xi[var + "_d2r1_" + t] = xi[var + '_' + t]
                    xi[var + "_d2r2_" + t] = xi[var + '_' + t]                         
                            
                            
    return xi
                        
                            
                            
def imput_df_nearest(data_clean, t_instances):
    
    imputed_df = data_clean.apply(imput_data_point, axis = 1, args=(t_instances,))   

    return imputed_df            
                            
                            