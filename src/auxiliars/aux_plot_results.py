# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:00:24 2025

@author: egarate
"""

import os
import pickle
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np


def extract_ti_from_filename(filename):
    match = re.search(r'n_(\d+)(\D+)(\d+)', filename)
    return int(match.group(3)) if match else None

def create_dataset_from_directory(directory):
    data = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as file:
                results = pickle.load(file)
                
                row = {
                    "model": results["model_type"],
                    "n_points": results["n_points"],
                    "t_dimension": results["t_dimension"],
                    "ti": extract_ti_from_filename(filename),
                    "t0": results["t_instances"][0] if "t_instances" in results and results["t_instances"] else None,
                    "tf": results["t_instances"][-2] if "t_instances" in results and results["t_instances"] else None
                }
                
                for key, value in results.get("comp_costs", {}).items():
                    if isinstance(value, list):
                        row[key] = sum(value)  
                    elif isinstance(value, (int, float)):
                        row[key] = value
                    else:
                        row[key] = 0  
                
                for key, value in results.get("model_metrics", {}).items():
                   if isinstance(value, list) and len(value) > 1:
                       row[key] = value[-1]  
                   elif isinstance(value, (int, float)):
                       row[key] = value  
                   else:
                       row[key] = None
                
                data.append(row)
    
    df = pd.DataFrame(data)
    
    return df

def add_computational_cost_sum(all_results_data):
    
    cost_columns = [col for col in all_results_data.columns if col not in ["model", "n_points", "t_dimension", "ti"]]
    all_results_data["total_comp_cost"] = all_results_data[cost_columns].fillna(0).sum(axis=1)    
    
    return all_results_data
    
def plot_computational_costs_n_samples_fixed(df, output_directory):
    
    os.makedirs(output_directory, exist_ok=True)
    models = ["XGB", "CNN", "CNN_self_supervised"]
    unique_n_points = df["n_points"].unique()
    colors = ['#FFE584', '#8EBD9D', '#FAD564']
    
    for n in unique_n_points:
        
        df_filtered = df[df["n_points"] == n]
        grouped_df = df_filtered.groupby(["model", "t_dimension"])['total_comp_cost'].mean().unstack(level=0)
        
        ax = grouped_df.plot(kind='bar', figsize=(10, 6), color=colors)
        plt.title(f'Computational Cost for n_points = {n}', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Instances (t_dimension)', fontsize=12, fontweight='bold')
        plt.ylabel('Average Computational Cost', fontsize=12, fontweight='bold')
        plt.legend(title='Model', fontsize=10)
        plt.xticks(rotation=0, fontsize=10)
        plt.yticks(fontsize=10)
        
        plot_filename = os.path.join(output_directory, f'plot_comp_costs_per_{n}_samples.pdf')
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close() 
    
def plot_computational_costs_t_dimension_fixed(df, output_directory):
    
    os.makedirs(output_directory, exist_ok=True)
    models = ["XGB", "CNN", "CNN_self_supervised"]
    unique_t_dimensions = df["t_dimension"].unique()
    colors = ['#F8B3CC', '#C7B65D', '#F34732']
    
    for t in unique_t_dimensions:
        df_filtered = df[df["t_dimension"] == t]
        grouped_df = df_filtered.groupby(["model", "n_points"])['total_comp_cost'].mean().unstack(level=0)
        
        ax = grouped_df.plot(kind='bar', figsize=(10, 6), color=colors)
        plt.title(f'Computational Cost for t_dimension = {t}', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Samples (n_points)', fontsize=12, fontweight='bold')
        plt.ylabel('Average Computational Cost', fontsize=12, fontweight='bold')
        plt.legend(title='Model', fontsize=10)
        plt.xticks(rotation=0, fontsize=10)
        plt.yticks(fontsize=10)
        
        plot_filename = os.path.join(output_directory, f'plot_comp_costs_per_{t}_t_dimension.pdf')
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()    
    
    
def plot_model_metrics_per_t0(df, metric_name, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    models = df["model"].unique()
    colors = {'XGB': '#5C6E6C', 'CNN_base': '#A6B7AA', 'CNN_self_supervised': '#D39D87'}
    
    plt.figure(figsize=(10, 6))
    t0_values = sorted(df["t0"].unique())
    positions = []
    boxplot_data = []
    colors_list = []
    
    for i, t0 in enumerate(t0_values):
        subset = df[df["t0"] == t0]
        models_present = subset["model"].unique()
        for j, model in enumerate(models):
            model_subset = subset[subset["model"] == model][metric_name].dropna().values
            if len(model_subset) > 0:
                boxplot_data.append(model_subset)
                positions.append(i * (len(models) + 1) + j)
                colors_list.append(colors.get(model, '#000000'))
    
    bplot = plt.boxplot(boxplot_data, positions=positions, patch_artist=True)
    
    for patch, color in zip(bplot['boxes'], colors_list):
        patch.set_facecolor(color)
    
    plt.xticks(ticks=np.arange(0, len(t0_values) * (len(models) + 1), (len(models) + 1)), labels=t0_values, fontsize=12, rotation=45)
    plt.title(f'Boxplot of {metric_name} grouped by Initial t0', fontsize=14, fontweight='bold')
    plt.xlabel('Initial t0', fontsize=14, fontweight='bold')
    plt.ylabel(metric_name, fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12)
    
    plt.legend([plt.Line2D([0], [0], color=color, lw=4) for color in colors.values()], models, loc='lower right', title='Model')
    
    plot_filename = os.path.join(output_directory, f'boxplot_{metric_name}_per_t0_all_models.pdf')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    

def plot_model_metrics_per_tf(df, metric_name, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    models = df["model"].unique()
    colors = {'XGB': '#5C6E6C', 'CNN_base': '#A6B7AA', 'CNN_self_supervised': '#D39D87'}
    
    plt.figure(figsize=(10, 6))
    tf_values = sorted(df["tf"].unique())
    positions = []
    boxplot_data = []
    colors_list = []
    
    for i, tf in enumerate(tf_values):
        subset = df[df["tf"] == tf]
        models_present = subset["model"].unique()
        for j, model in enumerate(models):
            model_subset = subset[subset["model"] == model][metric_name].dropna().values
            if len(model_subset) > 0:
                boxplot_data.append(model_subset)
                positions.append(i * (len(models) + 1) + j)
                colors_list.append(colors.get(model, '#000000'))
    
    bplot = plt.boxplot(boxplot_data, positions=positions, patch_artist=True)
    
    for patch, color in zip(bplot['boxes'], colors_list):
        patch.set_facecolor(color)
    
    plt.xticks(ticks=np.arange(0, len(tf_values) * (len(models) + 1), (len(models) + 1)), labels=tf_values, fontsize=12, rotation=45)
    plt.title(f'Boxplot of {metric_name} grouped by final instance', fontsize=14, fontweight='bold')
    plt.xlabel('Last training t', fontsize=14, fontweight='bold')
    plt.ylabel(metric_name, fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12)
    
    plt.legend([plt.Line2D([0], [0], color=color, lw=4) for color in colors.values()], models, loc='lower right', title='Model')
    
    plot_filename = os.path.join(output_directory, f'boxplot_{metric_name}_per_t0_all_models.pdf')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    
    
def plot_scatter_cost_vs_metric(df, metric_name, model, n_points, output_directory):
    
    os.makedirs(output_directory, exist_ok=True)
    df_filtered = df[(df["model"] == model) & (df["n_points"] == n_points)]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df_filtered["total_comp_cost"], df_filtered[metric_name], c='blue', label=f'{model}, n_points={n_points}')
    
    for _, row in df_filtered.iterrows():
        plt.text(row["total_comp_cost"], row[metric_name], str(row["ti"]), fontsize=9, ha='right', va='bottom')
    
    plt.xlabel("Computational Cost", fontsize=12, fontweight='bold')
    plt.ylabel(metric_name, fontsize=12, fontweight='bold')
    plt.title(f'Scatter plot of {metric_name} vs Computational Cost for {model}, n_points={n_points}', fontsize=14, fontweight='bold')
    plt.legend()
    
    plot_filename = os.path.join(output_directory, f'scatter_{metric_name}_vs_cost_{model}_n{n_points}.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()    
    
    
def plot_mean_r2_per_n_points(df, metric_name, model, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    df_filtered = df[df["model"] == model]
    mean_r2_per_n = df_filtered.groupby("n_points")[metric_name].mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_r2_per_n.index, mean_r2_per_n.values, marker='o', linestyle='-', color='b', label=f'Mean {metric_name} for {model}')
    
    plt.xlabel("Number of Samples (n_points)", fontsize=12, fontweight='bold')
    plt.ylabel(f"Mean {metric_name}", fontsize=12, fontweight='bold')
    plt.title(f"Mean {metric_name} vs. Number of Samples for {model}", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True)
    
    plot_filename = os.path.join(output_directory, f'mean_{metric_name}_vs_n_points_{model}.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    
    
    
    
    
    
    
    
    
    
    