# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: optimar_env
#     language: python
#     name: python3
# ---

# %%
import os
import sys
from pathlib import Path

def find_git_root(path=None):
    if path is None:
        try:
            path = Path(__file__).resolve()
        except NameError:
            path = Path.cwd()
    else:
        path = Path(path).resolve()

    for parent in [path] + list(path.parents):
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError("No .git directory found in any parent")

PROJ_ROOT = find_git_root()
print("PROJ_ROOT:", PROJ_ROOT)
sys.path.insert(0, str(PROJ_ROOT))
os.chdir(PROJ_ROOT)

# %%
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# %%
def read_csvs(dir_path):
    todos_los_csv = []
    for subcarpeta, _, archivos in os.walk(dir_path):
        for archivo in archivos:
            if archivo.endswith(".csv"):
                ruta_completa = os.path.join(subcarpeta, archivo)
                try:
                    df = pd.read_csv(ruta_completa)
                    df["subcarpeta"] = os.path.basename(subcarpeta)
                    df["csv_filename"] = archivo
                    todos_los_csv.append(df)
                except Exception as e:
                    print(f"Error leyendo {ruta_completa}: {e}")

    df_csv_combinado = pd.concat(todos_los_csv, ignore_index=True)
    return df_csv_combinado


# %%
def fit_and_print_linear_models(df, y_var, saving_folder, variables_label, save_pdf = False):
    X = df[['duration']]
    y = df[[y_var]]
    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    r2 = r2_score(y, y_pred)
    # Extraer coeficiente e intercepto
    m = modelo.coef_[0][0]
    b = modelo.intercept_[0]

    # Mostrar resultados
    print(f"\n📊 {y_var} ~ duration")
    print(f"  Fórmula: ŷ = {m:.8f}·x + {b:.8f}")
    print(f"  Coeficiente (pendiente): {m:.8f}")
    print(f"  Intercepto: {b:.8f}")
    print(f"  R²: {r2:.10f}")
    # Tamaños y estilo
    title_size = 26
    label_size = 22
    tick_size = 24
    legend_size = 24

    plt.figure(figsize=(10, 6))
    plt.plot(df['duration'], y_pred, color='orange', label="Lineal Regression", lw=5, zorder=1) 
    sns.scatterplot(x='duration', y=y_var, data=df, label="Data", s=70, zorder=2)
    # plt.title(f'{variables_label} vs Duration (s)', fontsize=title_size)
    plt.xlabel('Time (s)', fontsize=label_size)
    plt.ylabel(variables_label, fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.legend(fontsize=legend_size)
    plt.grid(True)
    plt.tight_layout()
    if save_pdf:
        filename = os.path.join(saving_folder, f"codecarbon_{y_var}_vs_duration.pdf")
        plt.savefig(filename, format='pdf', bbox_inches='tight')

    plt.show()
    plt.close()


# %%
def plot_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Mapa de calor de correlaciones")
    plt.tight_layout()
    plt.show()


# %%
def emissions_rate_summary_and_plot(df):
    # Filter Sweden and get csv_filename
    df_sweden = df[df['country_name'] == 'Sweden'].copy()
    df_sweden['csv_prefix'] = df_sweden['csv_filename'].str.split('_').str[0]

    # Mean values
    media_por_mode = df_sweden.groupby('mode')['emissions_rate'].mean().reset_index()
    media_por_csv = df_sweden.groupby('csv_prefix')['emissions_rate'].mean().reset_index()

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar plot mean per mode
    sns.barplot(x='mode', y='emissions_rate', data=media_por_mode, ax=axes[0], palette="Blues_d")
    axes[0].set_title('Media de emissions_rate por mode')
    axes[0].set_ylabel('Media emissions_rate')
    axes[0].set_xlabel('Mode')
    for i, row in media_por_mode.iterrows():
        axes[0].text(i, row['emissions_rate'], f"{row['emissions_rate']:.8e}", 
                     ha='center', va='bottom', fontsize=9)

    # Bar plot mean per csv_prefix
    sns.barplot(x='csv_prefix', y='emissions_rate', data=media_por_csv, ax=axes[1], palette="Greens_d")
    axes[1].set_title('Media de emissions_rate por csv_prefix')
    axes[1].set_ylabel('Media emissions_rate')
    axes[1].set_xlabel('CSV Prefix')
    axes[1].tick_params(axis='x', rotation=45)
    for i, row in media_por_csv.iterrows():
        axes[1].text(i, row['emissions_rate'], f"{row['emissions_rate']:.8e}", 
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    return media_por_mode, media_por_csv


# %%
ruta_classic = "data/outputs/classic/results/article-results/codecarbon"
ruta_adaptive = "data/outputs/adaptive/results/final_results/codecarbon"
saving_folder = "data/outputs/codecarbon_plots"
os.makedirs(saving_folder, exist_ok=True)

df_classic = read_csvs(ruta_classic)
df_classic["mode"] = "classic"

df_adaptive = read_csvs(ruta_adaptive)
df_adaptive["mode"] = "adaptive"

df_total = pd.concat([df_classic, df_adaptive], ignore_index=True)

# %%
indice_canada = df_total[df_total['country_name'] == 'Canada'].index[0]

# %%
variables_objetivo = ['emissions', 'cpu_energy', 'ram_energy', 'energy_consumed']
variables_label = ['CO₂ Emissions (kg)', 'CPU Energy (kWh)', 'RAM Energy (kWh)', 'Energy Consumed (kWh)']   
saving_pdfs = [True, False, False, True]
for i, y_var in enumerate(variables_objetivo):
    fit_and_print_linear_models(df_total.drop(indice_canada), y_var, saving_folder, variables_label[i], save_pdf = saving_pdfs[i])

# %%
plot_correlation_heatmap(df_total)

# %%
media_mode, media_csv = emissions_rate_summary_and_plot(df_total)

# %%
for col in variables_objetivo:
    rate_col = f"{col}_rate"
    df_total[rate_col] = df_total[col] / df_total['duration']
    media = df_total[rate_col].mean()
    print(f"🔍 Media de {rate_col}: {media:.8e} por segundo")

# %%
# Variables
X = df_total[['duration']]
y = df_total[['emissions']]

# Fit model
modelo = LinearRegression()
modelo.fit(X, y)
y_pred = modelo.predict(X)

# Absolute errors
residuos = np.abs(y.values.flatten() - y_pred.flatten())

# Highest error
indice_max_error = residuos.argmax()

# %%
df_total.iloc[indice_max_error]

# %%
df_total = df_total.reset_index(drop=True)


# %%
df_total

# %%
indice_canada

# %%
df_total.iloc[indice_canada-1:indice_canada+2]

# %%
df_total = df_total.sort_values(by="timestamp")

# %%
regiones = df_total['country_name'].unique()
print("📍 Regiones únicas:", regiones)

# %%
conteo_regiones = df_total['country_name'].value_counts()
print(conteo_regiones)
