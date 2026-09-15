# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:32:14 2025

@author: egarate
"""

from src.auxiliars.aux_analyze_metrics import *

adaptive_directory = "//datastore/ia/data-analytics/ADAPTIVE-SAMPLING/outputs/adaptive/results/final_results"

classic_directory = "//datastore/ia/data-analytics/ADAPTIVE-SAMPLING/outputs/classic/results/article-results"

saving_path = "//datastore/ia/data-analytics/ADAPTIVE-SAMPLING/plots"

plot_metric_by_model_per_tinstance(adaptive_directory, classic_directory, metric_name="metric", saving_path=saving_path)

plot_avg_cost_vs_metric_by_model_and_ti(adaptive_directory, classic_directory, metric_name="metric", saving_path=saving_path)

plot_cost_vs_metric_by_model_all_ti(adaptive_directory, classic_directory, metric_name="metric", modelos=None, n_samples = 150, saving_path=saving_path)

plot_cost_vs_metric_by_model_all_ti(adaptive_directory, classic_directory, metric_name="metric", modelos=None, n_samples = 300, saving_path=saving_path)

plot_cost_vs_metric_by_model_all_ti(adaptive_directory, classic_directory, metric_name="metric", modelos=None, n_samples = 500, saving_path=saving_path)

plot_cost_vs_metric_by_model_all_ti(adaptive_directory, classic_directory, metric_name="metric", modelos=None, n_samples = 700, saving_path=saving_path)
