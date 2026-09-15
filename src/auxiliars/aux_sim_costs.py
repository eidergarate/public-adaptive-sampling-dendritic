# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 15:38:47 2025

@author: egarate
"""

from src.auxiliars.aux_analyze_comp_costs import *

import seaborn as sns
import matplotlib.pyplot as plt

def get_density_plot_sim(dir_adaptive, dir_classic, saving_path):
    
    df_vect = extract_vectorized_results(dir_adaptive)
    df_flat = extract_results_from_directory(dir_classic)

    df_all = pd.concat([df_vect, df_flat], ignore_index=True)
        
    df_all['t_sim_scaled'] = df_all["t_sim"]/df_all["n_points"]
    
    media = df_all['t_sim_scaled'].mean()
    median = df_all['t_sim_scaled'].median()
    
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df_all['t_sim_scaled'])  
    plt.axvline(media, color='red', linestyle='--', label=f"Media = {media:.2f}")
    plt.axvline(median, color='green', linestyle='--', label=f"Media = {media:.2f}")

    plt.title("Density", fontsize = 17)
    plt.xlabel("Simulation time (s)", fontsize = 16)
    plt.ylabel("")
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plot_name = saving_path + "/simulation_cost.pdf"
    plt.savefig(plot_name, format='pdf', bbox_inches='tight')
    
    plt.show()