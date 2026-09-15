# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:35:31 2025

@author: egarate
"""

from src.auxiliars.aux_analyze_comp_costs import *
from src.auxiliars.aux_sim_costs import *

adaptive_directory = "//datastore/ia/data-analytics/ADAPTIVE-SAMPLING/outputs/adaptive/results/final_results"

classic_directory = "//datastore/ia/data-analytics/ADAPTIVE-SAMPLING/outputs/classic/results/article-results"

saving_path = "//datastore/ia/data-analytics/ADAPTIVE-SAMPLING/plots"

plot_computational_costs_line(adaptive_directory, classic_directory, cost_column="t_training", saving_path=saving_path)

plot_computational_costs_line(adaptive_directory, classic_directory, cost_column="t_new_sampling", saving_path=saving_path)

plot_computational_costs_line(adaptive_directory, classic_directory, cost_column="total_cost", saving_path=saving_path)

plot_computational_costs_line(adaptive_directory, classic_directory, cost_column="t_sim", saving_path=saving_path)

plot_computational_costs_line(adaptive_directory, classic_directory, cost_column="t_sampling", saving_path=saving_path)

plot_costs_by_model(adaptive_directory, classic_directory, cost_column="t_training", saving_path=saving_path)

get_density_plot_sim(adaptive_directory, classic_directory, saving_path)
