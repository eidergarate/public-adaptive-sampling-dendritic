# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 11:36:23 2025

@author: egarate
"""

from src.auxiliars.aux_analyze_balanced_results import *

adaptive_directory = "//datastore/ia/data-analytics/ADAPTIVE-SAMPLING/outputs/adaptive/results/final_results"

classic_directory = "//datastore/ia/data-analytics/ADAPTIVE-SAMPLING/outputs/classic/results/article-results"

saving_path = "//datastore/ia/data-analytics/ADAPTIVE-SAMPLING/plots"

plot_training_and_sampling_costs_comparison(adaptive_directory, classic_directory, "metric", saving_path)