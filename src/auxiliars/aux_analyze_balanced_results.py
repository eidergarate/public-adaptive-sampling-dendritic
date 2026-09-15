# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 09:45:13 2025

@author: egarate
"""

from src.auxiliars.aux_analyze_metrics import *
import pandas as pd
from src.auxiliars.aux_analyze_comp_costs import *


def get_when_better_than_r2(dir_adaptive, dir_classic, min_r2 = 0.85):
    
    df_vect = extract_vectorized_results_with_metrics(dir_adaptive)
    df_flat = extract_results_from_directory_with_metrics(dir_classic)
    df = pd.concat([df_vect, df_flat], ignore_index=True)
    
    resultados = []

    # group by models and temporal instances sets
    for (model, t_instance), group in df.groupby(["model", "t_instance"]):
        group_valid = group[group["metric"] >= min_r2]

        if not group_valid.empty:
            min_n = group_valid.sort_values("n_points")["n_points"].iloc[0]
        else:
            min_n = 700 + 1  

        resultados.append({
            "model": model,
            "t_instance": t_instance,
            "n_min_points_for_metric": min_n
        })

    return pd.DataFrame(resultados)
    
    
def summarize_klasikoa_vs_moldagarria(dir_adaptive, dir_classic, metric_name="metric"):

    df_vect = extract_vectorized_results_with_metrics(dir_adaptive)
    df_flat = extract_results_from_directory_with_metrics(dir_classic)
    df = pd.concat([df_vect, df_flat], ignore_index=True)

    required_cols = {"model", "t_instance", "n_points", metric_name, "t_training", "t_sampling"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Faltan columnas requeridas en el DataFrame: {required_cols - set(df.columns)}")

    # Naming for plots
    base_models = {
        "XGB": "XGB_stochastic",
        "CNN": "CNN_MC_dropout",
        "CNN_ss": "CNN_MC_dropout_ss"
    }

    filas = []

    for tipo_modelo, modelo_estocastico in base_models.items():
        df_base_all = df[df["model"] == tipo_modelo]
        df_stoc_all = df[df["model"] == modelo_estocastico]

        for t_i in df_base_all["t_instance"].unique():
            df_base = df_base_all[df_base_all["t_instance"] == t_i]
            df_stoc = df_stoc_all[df_stoc_all["t_instance"] == t_i]

            if df_base.empty:
                continue

            fila_max = df_base.loc[df_base[metric_name].idxmax()]
            max_klasikoa = fila_max[metric_name]
            n_points_klasikoa = fila_max["n_points"]
            t_ent_k = fila_max["t_training"]
            t_lag_k = fila_max["t_sampling"]

            df_stoc_valid = df_stoc[df_stoc[metric_name] >= max_klasikoa]
            if not df_stoc_valid.empty:
                fila_mold = df_stoc_valid.sort_values("n_points").iloc[0]
            else:
                fila_mold = df_stoc.sort_values("n_points", ascending=False).iloc[0]

            n_points_moldagarria = fila_mold["n_points"]
            t_ent_m = fila_mold["t_training"]
            t_lag_m = fila_mold["t_sampling"]

            filas.append({
                "model_type": tipo_modelo,
                "t_instance": t_i,
                "max_classic": max_klasikoa,
                "n_points_classic": n_points_klasikoa,
                "t_training_classic": t_ent_k,
                "t_sampling_classic": t_lag_k,
                "n_points_adaptive": n_points_moldagarria,
                "t_training_adaptive": t_ent_m,
                "t_sampling_adaptive": t_lag_m
            })

    return pd.DataFrame(filas)


def get_data_vect_for_lm(directory):
    
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

            codecarbon_results = data.get("CO2_emissions_kg", {})
            total_emisiones = sum(codecarbon_results.values())

            for i in range(length):
                n = n_points_list[i]
                t_training = corrected_training[i]
                t_sim = t_sim_list[i]
                t_doe = t_doe_list[i]
                t_sampling = t_doe_list[i] + t_sim_list[i]
                total_cost = t_doe_list[i] + t_sim_list[i] + t_training

                rows.append({
                    "model": model,
                    "t_instance": mapping.get(f"t{t_instance}", f"t{t_instance}"),
                    "n_points": n,
                    "t_new_sampling": t_sampling,
                    "t_training": t_training,
                    "t_sampling": t_doe,
                    "t_sim": t_sim,
                    "total_cost": total_cost,
                    "co2_kg_total": total_emisiones
                })

    return pd.DataFrame(rows)
    
import statsmodels.api as sm

def lm_costs_by_model_and_ti(dir_adaptive, dir_classic, cost_type):
    df_classic = extract_results_from_directory(dir_classic)
    df_adaptive = get_data_vect_for_lm(dir_adaptive)
    df = pd.concat([df_classic, df_adaptive], ignore_index=True)

    if cost_type not in df.columns or 'n_points' not in df.columns:
        raise ValueError("El DataFrame debe contener 'n_points' y el tipo de coste indicado.")

    resultados = []

    for (modelo, t_i), grupo in df.groupby(["model", "t_instance"]):
        if len(grupo) < 2:
            continue

        X = sm.add_constant(grupo["n_points"])
        y = grupo[cost_type]

        modelo_lm = sm.OLS(y, X).fit()

        coef = modelo_lm.params["n_points"]
        intercepto = modelo_lm.params["const"]
        r2 = modelo_lm.rsquared
        pval = modelo_lm.pvalues["n_points"]

        resultados.append({
            "modelo": modelo,
            "t_instance": t_i,
            "R2": r2,
            "coeficiente": coef,
            "intercepto": intercepto,
            "p_value": pval
        })

    results_df = pd.DataFrame(resultados)
    
    return results_df

def compute_costs_from_lm_models(dir_adaptive, dir_classic, metric_name="R2"):
    df_lm_entren = lm_costs_by_model_and_ti(dir_adaptive, dir_classic, cost_type="t_training")
    df_lm_lagin = lm_costs_by_model_and_ti(dir_adaptive, dir_classic, cost_type="t_sampling")

    df_refs = summarize_klasikoa_vs_moldagarria(dir_adaptive, dir_classic, metric_name=metric_name)

    resultados = []

    for _, row in df_refs.iterrows():
        modelo_base = row["model_type"]
        modelo_estok = {
            "XGB": "XGB_stochastic",
            "CNN": "CNN_MC_dropout_ss",
            "CNN_ag": "CNN_MC_dropout_ss"
        }[modelo_base]

        t_i = row["t_instance"]
        n_klasikoa = row["n_points_classic"]
        n_molda = row["n_points_adaptive"]
        
        if np.isnan(n_molda):
            n_molda = 700

        def get_coef(df_lm, modelo, t_i):
            fila = df_lm[(df_lm["modelo"] == modelo) & (df_lm["t_instance"] == t_i)]
            if not fila.empty:
                return fila.iloc[0]["coeficiente"], fila.iloc[0]["intercepto"]
            else:
                return None, None

        a_entr_k, b_entr_k = get_coef(df_lm_entren, modelo_base, t_i)
        a_entr_m, b_entr_m = get_coef(df_lm_entren, modelo_estok, t_i)

        a_lag_k, b_lag_k = get_coef(df_lm_lagin, modelo_base, t_i)
        a_lag_m, b_lag_m = get_coef(df_lm_lagin, modelo_estok, t_i)

        if None in [a_entr_k, b_entr_k, a_lag_k, b_lag_k, a_entr_m, b_entr_m, a_lag_m, b_lag_m]:
            continue  

        kostea_klasikoa = n_klasikoa * a_entr_k + b_entr_k + n_klasikoa * a_lag_k + b_lag_k
        kostea_moldagarria = (
            n_molda * a_entr_m + b_entr_m + n_molda * a_lag_m + b_lag_m
            if pd.notna(n_molda) else None
        )
        
        kostea_klasikoa_neurtua = row['t_training_classic'] + row['t_sampling_classic']
        kostea_moldagarria_neurtua = row['t_training_adaptive'] + row['t_sampling_adaptive']

        resultados.append({
            "tipo_modelo": modelo_base,
            "t_instance": t_i,
            "n_points_classic": n_klasikoa,
            "n_points_daptive": n_molda,
            "approx_cost_classic": kostea_klasikoa,
            "approx_cost_adaptive": kostea_moldagarria,
            "classic_cost": kostea_klasikoa_neurtua,
            "adaptive_cost": kostea_moldagarria_neurtua,
        })

    approximated_costs = pd.DataFrame(resultados)
    
    return df_lm_entren, df_lm_lagin, df_refs, approximated_costs

    
def plot_training_and_sampling_costs_comparison(dir_adaptive, dir_classic, metric_name="metric", saving_path = ""):
    df_refs = summarize_klasikoa_vs_moldagarria(dir_adaptive, dir_classic, metric_name=metric_name)
    df_refs['etiqueta'] = df_refs['model_type'] + " - " + df_refs['t_instance']
    x = df_refs['etiqueta']
    x_pos = range(len(x))
    width = 0.35

    color_map = {
        "XGB": "#1f77b4",
        "CNN": "#2ca02c",
        "CNN_ss": "#ff7f0e",
    }

    title_size = 16
    label_size = 15
    tick_size = 14
    legend_size = 14

    plt.figure(figsize=(12, 5))
    for i, row in df_refs.iterrows():
        color = color_map.get(row["model_type"], "gray")
        plt.bar(i - width/2, row["t_training_classic"], width,
                color=color, alpha=0.4, label='Classic Trainig' if i == 0 else "")
        plt.bar(i + width/2, row["t_training_adaptive"], width,
                color=color, alpha=1.0, label='Adaptive Training' if i == 0 else "")

    plot_training_name = saving_path + "/entrenamendua_same_cost.pdf"
    
    plt.ylabel("Time (s)", fontsize=label_size)
    plt.title("", fontsize=title_size)
    plt.xticks(x_pos, x, rotation=45, ha='right', fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.legend(fontsize=legend_size)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(plot_training_name, format='pdf', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 5))
    for i, row in df_refs.iterrows():
        color = color_map.get(row["model_type"], "gray")
        plt.bar(i - width/2, row["t_sampling_classic"], width,
                color=color, alpha=0.4, label='Sampling Cl.' if i == 0 else "")
        plt.bar(i + width/2, row["t_sampling_adaptive"], width,
                color=color, alpha=1.0, label='Sampling Ad.' if i == 0 else "")

    plot_sampling_name = saving_path + "/sampling_same_cost.pdf"    

    plt.ylabel("Time (s)", fontsize=label_size)
    plt.title("", fontsize=title_size)
    plt.xticks(x_pos, x, rotation=45, ha='right', fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.legend(fontsize=legend_size)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(plot_sampling_name, format='pdf', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 5))
    for i, row in df_refs.iterrows():
        color = color_map.get(row["model_type"], "gray")
        plt.bar(i - width/2, row["n_points_classic"], width,
                color=color, alpha=0.4, label='n classic' if i == 0 else "")
        plt.bar(i + width/2, row["n_points_adaptive"], width,
                color=color, alpha=1.0, label='n adaptive' if i == 0 else "")
        
    plot_n_sampling_name = saving_path + "/n_sampling_same_cost.pdf"        

    plt.ylabel("Sampling size", fontsize=label_size)
    plt.title("", fontsize=title_size)
    plt.xticks(x_pos, x, rotation=45, ha='right', fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.legend(fontsize=legend_size)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(plot_n_sampling_name, format='pdf', bbox_inches='tight')
    plt.show()