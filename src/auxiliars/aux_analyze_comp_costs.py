# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 10:18:22 2025

@author: egarate
"""

import os
import pickle
import pandas as pd

def extract_results_from_directory(directory):
    records = []
    mapping = {"t3": "ID1", "t6": "ID2", "t9": "ID3"}
    model_mapping = {
        "CNN_MC_dropout": "CNN_MC_dropout",
        "XGB_probabilistic": "XGB_stochastic",
        "CNN_MC_dropout_ss": "CNN_MC_dropout_ss",
        "CNN_base": "CNN",
        "CNN_self_supervised": "CNN_ss",
        "XGB": "XGB"
    }

    for filename in os.listdir(directory):
        if filename.endswith(".pkl") and "results" in filename:
            t_instance = filename.split("t")[-1].replace(".pkl", "")
            filepath = os.path.join(directory, filename)

            with open(filepath, "rb") as f:
                data = pickle.load(f)

            model = data.get("model_type", "Unknown")
            model = model_mapping.get(model, model)
            n_points = data.get("n_points", None)

            comp_costs = data.get("comp_costs", {})
            t_doe = comp_costs.get("t_doe", 0)
            t_sim = comp_costs.get("t_simulation", 0)
            t_train = comp_costs.get("t_training", 0)

            codecarbon_results = data.get("CO2_emissions_kg", {})
            total_emisiones = sum(codecarbon_results.values())
            
            total_cost = t_doe + t_sim + t_train
            t_sampling = t_doe + t_sim

            records.append({
                "model": model,
                "t_instance": f"t{t_instance}",
                "n_points": n_points,
                "t_new_sampling": t_sampling,
                "t_training": t_train,
                "t_sampling": t_doe,
                "t_sim": t_sim,
                "total_cost": total_cost,
                "co2_kg_total": total_emisiones
            })

    df = pd.DataFrame(records)

    df = df[df["t_instance"].isin(["t3", "t6", "t9"])]
    df = df[df['n_points'] <= 700]
    df["t_instance"] = df["t_instance"].map(mapping)

    return df

def extract_vectorized_results(directory):
    rows = []
    mapping = {"t3": "ID1", "t6": "ID2", "t9": "ID3"}
    model_mapping = {
        "CNN_MC_dropout": "CNN_MC_dropout",
        "XGB_probabilistic": "XGB_stochastic",
        "CNN_MC_dropout_ss": "CNN_MC_dropout_ss",
        "CNN_base": "CNN",
        "CNN_self_supervised": "CNN_ss",
        "XGB": "XGB"
    }

    for filename in os.listdir(directory):
        if filename.endswith(".pkl") and "results" in filename:
            t_instance = filename.split("t")[-1].replace(".pkl", "")
            filepath = os.path.join(directory, filename)

            with open(filepath, "rb") as f:
                data = pickle.load(f)

            model = data.get("model_type", "Unknown")
            model = model_mapping.get(model, model)
            n_points_list = data.get("n_samples_list", [])
            comp_costs = data.get("comp_costs", {})

            t_doe_list = comp_costs.get("t_doe", [])
            t_sim_list = comp_costs.get("t_simulation", [])
            t_train_list = comp_costs.get("t_training", [])

            length = min(len(n_points_list), len(t_doe_list), len(t_sim_list), len(t_train_list))
            n_points_list = n_points_list[:length]
            t_doe_list = t_doe_list[:length]
            t_sim_list = t_sim_list[:length]
            t_train_list = t_train_list[:length]

            corrected_training = t_train_list.copy()
            for i in range(1, length - 1):
                prev = corrected_training[i - 1]
                curr = corrected_training[i]
                next_ = corrected_training[i + 1]
                if not (prev <= curr <= next_ * 1.5):
                    x0, x1, x2 = n_points_list[i - 1], n_points_list[i], n_points_list[i + 1]
                    weight = (x1 - x0) / (x2 - x0) if (x2 - x0) != 0 else 0.5
                    corrected_training[i] = prev + (next_ - prev) * weight

            t_sampling_cumulative = []
            t_doe_cumulative = []
            t_sim_cumulative = []
            t_train_cumulative = []
            sampling_cumulative = 0
            doe_cumulative = 0
            sim_cumulative = 0
            train_cumulative = 0
            for i in range(length):
                doe_cumulative += t_doe_list[i]
                sim_cumulative += t_sim_list[i]
                sampling_cumulative += t_doe_list[i] + t_sim_list[i]
                train_cumulative += corrected_training[i]
                
                t_sampling_cumulative.append(sampling_cumulative)
                t_doe_cumulative.append(doe_cumulative)
                t_sim_cumulative.append(sim_cumulative)
                t_train_cumulative.append(train_cumulative)

            for i in range(length):
                n = n_points_list[i]
                t_sampling = t_sampling_cumulative[i]
                t_training = t_train_cumulative[i]
                t_sim = t_sim_cumulative[i]
                t_doe = t_doe_cumulative[i]
                total_cost = t_doe_list[i] + t_sim_list[i] + t_training

                rows.append({
                    "model": model,
                    "t_instance": mapping.get(f"t{t_instance}", f"t{t_instance}"),
                    "n_points": n,
                    "t_new_samples": t_sampling,
                    "t_training": t_training,
                    "t_sampling": t_doe,
                    "t_sim": t_sim,
                    "total_cost": total_cost
                })

    return pd.DataFrame(rows)


import matplotlib.pyplot as plt

def plot_computational_costs_line(dir_adaptive, dir_classic, cost_column, saving_path):
    df_vect = extract_vectorized_results(dir_adaptive)
    df_flat = extract_results_from_directory(dir_classic)

    df_all = pd.concat([df_vect, df_flat], ignore_index=True)

    if cost_column not in df_all.columns:
        raise ValueError(f"'{cost_column}' no es una columna válida. Usa: 't_sampling', 't_training', 'total_cost', 't_simulation', 't_doe'.")

    color_map = {
        "XGB": "#1f77b4",  
        "XGB_stochastic": "#1f77b4",
        "CNN": "#2ca02c",  
        "CNN_MC_dropout": "#2ca02c",
        "CNN_ss": "#ff7f0e",  
        "CNN_MC_dropout_ss": "#ff7f0e"
    }

    alpha_map = {
        "XGB": 0.4,
        "XGB_stochastic": 1.0,
        "CNN": 0.4,
        "CNN_MC_dropout": 1.0,
        "CNN_ss": 0.4,
        "CNN_MC_dropout_ss": 1.0
    }

    title_size = 16
    label_size = 16
    tick_size = 14
    legend_size = 14
    for t in sorted(df_all["t_instance"].unique()):
        
        plot_name = saving_path + "/" + cost_column + "_cost_" + t + ".pdf"
        df_t = df_all[df_all["t_instance"] == t]

        plt.figure(figsize=(10, 6))

        for model in df_t["model"].unique():
            df_model = df_t[df_t["model"] == model].sort_values("n_points")
            color = color_map.get(model, "gray")
            alpha = alpha_map.get(model, 1.0)
            plt.plot(df_model["n_points"], df_model[cost_column],
                     marker="o", label=model, color=color, alpha=alpha)

        plt.title("", fontsize=title_size)
        plt.xlabel("Sampling size", fontsize=label_size)
        plt.ylabel("Time (s)", fontsize=label_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.legend(title="Model", fontsize=legend_size, title_fontsize=legend_size)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_name, format='pdf', bbox_inches='tight')
        plt.show()



def plot_costs_by_model(dir_adaptive, dir_classic, cost_column, saving_path):

    title_size = 16
    label_size = 16
    tick_size = 14
    legend_size = 14

    df_vect = extract_vectorized_results(dir_adaptive)
    df_flat = extract_results_from_directory(dir_classic)

    df_all = pd.concat([df_vect, df_flat], ignore_index=True)

    if cost_column not in df_all.columns:
        raise ValueError(f"'{cost_column}' no es una columna válida. Usa: 't_sampling', 't_training', 'total_cost', 't_simulation', 't_doe'.")

    for model in sorted(df_all["model"].unique()):
        
        plot_name = saving_path + "/" + cost_column + "_cost_" + model + ".pdf"
        
        df_model = df_all[df_all["model"] == model]

        plt.figure(figsize=(10, 6))
        for t in sorted(df_model["t_instance"].unique()):
            df_t = df_model[df_model["t_instance"] == t].sort_values("n_points")
            plt.plot(df_t["n_points"], df_t[cost_column], marker="o", label=t)

        if model == "CNN_ss":
            izena = "Self supervised convolutional NN"
        elif model =="CNN":
            izena = "Convolutional NN"
        elif model == "CNN_MC_dropout":
            izena = "Adaptive convolutional NN"
        elif model == "CNN_MC_dropout_ss":
            izena = "Self supervised adaptive convolutional NN"
        elif model == "XGB":
            izena = "XGBoost"
        elif model == "XGB_stochastic":
            model = "Adaptive XGBoost"

        plt.title(f"{izena}", fontsize=title_size)
        plt.xlabel("Sampling size", fontsize=label_size)
        plt.ylabel("Time (s)", fontsize=label_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.legend(title="experiment", fontsize=legend_size, title_fontsize=legend_size)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_name, format='pdf', bbox_inches='tight')
        plt.show()

