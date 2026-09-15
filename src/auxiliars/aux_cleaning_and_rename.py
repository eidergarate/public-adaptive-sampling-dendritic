# -*- coding: utf-8 -*-
"""
Created on 17.07.24

@author: mgomez
"""

import re 

def cleaning_and_rename(df):

    df.insert(0, 'exp_id_new', df['exp_id'])

    df.drop(['exp_id'], axis = 1, inplace = True)
    df_clean= df.rename(columns={'exp_id_new': 'exp_id'})

    nuevos_nombres = []

    for columna in df_clean.columns:
        nuevo_nombre = columna
        nuevo_nombre = re.sub(r'_pos', '', nuevo_nombre)
        nuevo_nombre = re.sub(r'up', 'u', nuevo_nombre)
        nuevo_nombre = re.sub(r'right', 'r', nuevo_nombre)
        nuevo_nombre = re.sub(r'left', 'l', nuevo_nombre)
        nuevo_nombre = re.sub(r'down', 'd', nuevo_nombre)
        nuevo_nombre = re.sub(r'-3000', '1000', nuevo_nombre)
        nuevo_nombre = re.sub(r'-2000', '2000', nuevo_nombre)    
        nuevo_nombre = re.sub(r'\(xi, yi\)', '', nuevo_nombre)
        nuevo_nombre = re.sub(r'_0', '_4000', nuevo_nombre)
        nuevo_nombre = re.sub(r'__', '_', nuevo_nombre)
        nuevos_nombres.append(nuevo_nombre)

    df_clean_renamed = df_clean

    df_clean_renamed.rename(columns=dict(zip(df_clean.columns, nuevos_nombres)), inplace=True)

    df_clean_renamed = df_clean_renamed.sort_values(by=['exp_id', 'x', 'y'])

    df_clean_renamed = df_clean_renamed.reset_index(drop=True)

    column_names = df_clean_renamed.columns.tolist()
    column_names.insert(1, column_names.pop(column_names.index('theta0')))
    df_clean_renamed = df_clean_renamed[column_names]

    return(df_clean_renamed)


def format_ids(ids):
    """
    Formatea los IDs para su escritura en un archivo de texto.
    
    Parameters:
        ids (int, list of int): Un entero o una lista de enteros.
        
    Returns:
        str: La representación en cadena de los IDs.
    """
    if isinstance(ids, list):
        return ", ".join(map(str, ids))
    else:
        return str(ids)
