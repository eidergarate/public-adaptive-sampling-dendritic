# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:42:21 2025

@author: egarate
"""
import os
import pickle
import pandas as pd
import numpy as np

def extract_results_from_directory_with_metrics(directory):
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
            metrics = data.get("model_metrics", {})
            metrics = metrics['cv_metric']
            
            if isinstance(metrics, list):
                metrics = metrics[-1]

            t_doe = comp_costs.get("t_doe", 0)
            t_sim = comp_costs.get("t_simulation", 0)
            t_train = comp_costs.get("t_training", 0)

            codecarbon_results = data.get("CO2_emissions_kg", {})
            total_emisiones = sum(codecarbon_results.values())

            total_cost = t_doe + t_sim + t_train
            t_sampling = t_doe + t_sim
            
            row = {
                "model": model,
                "t_instance": f"t{t_instance}",
                "n_points": n_points,
                "t_new_sampling": t_sampling,
                "t_training": t_train,
                "t_sampling": t_doe,
                "t_sim": t_sim,
                "total_cost": total_cost,
                "metric": metrics,
                "co2_kg_total": total_emisiones
            }


            records.append(row)

    df = pd.DataFrame(records)

    df = df[df["t_instance"].isin(["t3", "t6", "t9"])]
    df = df[df['n_points'] <= 700]
    df["t_instance"] = df["t_instance"].map(mapping)
    
    return df


def extract_vectorized_results_with_metrics(directory):
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
            metrics = data.get("model_metrics", {})
            metrics = metrics['cv_metric']

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
            t_train_cumulative = []
            cumulative = 0
            train_cumulative = 0
            for i in range(length):
                cumulative += t_doe_list[i] + t_sim_list[i]
                train_cumulative += corrected_training[i]
                t_sampling_cumulative.append(cumulative)
                t_train_cumulative.append(train_cumulative)

            for i in range(length):
                row = {
                    "model": model,
                    "t_instance": mapping.get(f"t{t_instance}", f"t{t_instance}"),
                    "n_points": n_points_list[i],
                    "t_new_sampling": t_doe_list[i],
                    "t_sim": t_sim_list[i],
                    "t_sampling": t_sampling_cumulative[i],
                    "t_training": t_train_cumulative[i],
                    "total_cost": t_sampling_cumulative[i] + t_train_cumulative[i],
                    "metric":metrics[i]
                }

                rows.append(row)

    return pd.DataFrame(rows)


import matplotlib.pyplot as plt

def plot_metric_by_model_per_tinstance(dir_adaptive, dir_classic, metric_name, saving_path):
    import matplotlib.pyplot as plt
    import pandas as pd

    df_vect = extract_vectorized_results_with_metrics(dir_adaptive)
    df_flat = extract_results_from_directory_with_metrics(dir_classic)
    df = pd.concat([df_vect, df_flat], ignore_index=True)

    if metric_name not in df.columns:
        raise ValueError(f"La métrica '{metric_name}' no se encuentra en los resultados.")

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

    for t_instance in sorted(df["t_instance"].unique()):
        df_t = df[df["t_instance"] == t_instance]
        
        plot_name = saving_path + "/" + metric_name  + "_" + t_instance + ".pdf"

        plt.figure(figsize=(10, 6))
        for model in sorted(df_t["model"].unique()):
            df_model = df_t[df_t["model"] == model].sort_values("n_points")
            color = color_map.get(model, "gray")
            alpha = alpha_map.get(model, 1.0)
            plt.plot(
                df_model["n_points"],
                df_model[metric_name],
                marker="o",
                label=model,
                color=color,
                alpha=alpha
            )

        plt.title("", fontsize=title_size)
        plt.xlabel("Sampling size", fontsize=label_size)
        plt.ylabel("R squared", fontsize=label_size)
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.legend(
            title="Model",
            fontsize=legend_size,
            title_fontsize=legend_size,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,  
            frameon=False
            )
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_name, format='pdf', bbox_inches='tight')
        plt.show()



import seaborn as sns

def plot_avg_cost_vs_metric_by_model_and_ti(dir_adaptive, dir_classic, metric_name, saving_path):
    df_vect = extract_vectorized_results_with_metrics(dir_adaptive)
    df_flat = extract_results_from_directory_with_metrics(dir_classic)
    df = pd.concat([df_vect, df_flat], ignore_index=True)

    if metric_name not in df.columns or "total_cost" not in df.columns:
        raise ValueError(f"El DataFrame debe contener '{metric_name}' y 'total_cost'.")

    df_avg = df.groupby(["model", "t_instance"]).agg(
        avg_cost=("total_cost", "mean"),
        avg_metric=(metric_name, "mean")
    ).reset_index()

    plt.figure(figsize=(10, 7))
    sns.set(style="whitegrid")

    palette = sns.color_palette("tab10", df_avg["model"].nunique())

    sns.scatterplot(
        data=df_avg,
        x="avg_cost",
        y="avg_metric",
        hue="model",
        palette=palette,
        s=100
    )

    for _, row in df_avg.iterrows():
        plt.text(row["avg_cost"], row["avg_metric"], row["t_instance"],
                 fontsize=12, ha='left', va='center')

    plt.title(f"Mean computational costper ID and model")
    plt.xlabel("Mean computational cost")
    plt.ylabel(f"Mean R^2")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()
    
def plot_cost_vs_metric_by_model_all_ti(dir_adaptive, dir_classic, metric_name="metric", modelos=None, n_samples=None, saving_path=""):
    df_vect = extract_vectorized_results_with_metrics(dir_adaptive)
    df_flat = extract_results_from_directory_with_metrics(dir_classic)
    df = pd.concat([df_vect, df_flat], ignore_index=True)

    if metric_name not in df.columns or "total_cost" not in df.columns or "n_points" not in df.columns:
        raise ValueError("Faltan columnas requeridas: 'n_points', 'koste_totala', y/o la métrica.")

    if modelos is None:
        modelos = df["model"].unique().tolist()
    df = df[df["model"].isin(modelos)]

    if n_samples is not None:
        df_selected = []
        for (model, t_instance), group in df.groupby(["model", "t_instance"]):
            group = group.copy()
            group["distance"] = np.abs(group["n_points"] - n_samples)
            idx_min = group["distance"].idxmin()
            df_selected.append(df.loc[idx_min])
        df = pd.DataFrame(df_selected)

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
    label_size = 19
    tick_size = 16
    legend_size = 17
    point_label_size = 14  

    plt.figure(figsize=(12, 7))
    sns.set(style="whitegrid")

    for model in df["model"].unique():
        df_m = df[df["model"] == model]
        plt.scatter(
            df_m["total_cost"],
            df_m[metric_name],
            label=model,
            color=color_map.get(model, "gray"),
            alpha=alpha_map.get(model, 1.0),
            s=80
        )

    for _, row in df.iterrows():
        plt.text(row["total_cost"], row[metric_name], row["t_instance"],
                 fontsize=point_label_size, ha='left', va='center')

    plot_name = saving_path + "/" + "compared_metric_n_" + str(n_samples) + ".pdf"
    
    plt.title("", fontsize=title_size)
    plt.xlabel("Total computational cost", fontsize=label_size)
    plt.ylabel("R squared", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.legend(
        title="Model",
        fontsize=legend_size,
        title_fontsize=legend_size,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False
    )
    plt.tight_layout()
    plt.savefig(plot_name, format='pdf', bbox_inches='tight')
    plt.show()
