# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 11:39:49 2024

@author: egarate
"""

import numpy as np
import pandas as pd

from src.auxiliars.aux_compute_phip_n_k_LHS import *
from src.auxiliars.aux_create_random_n_k_LHS import *
from src.auxiliars.aux_LHS_distances import *


def get_population(n_points, n_features, n_pop):
    
    population = pd.DataFrame(
        np.zeros((n_points * n_pop, n_features + 1)),
        columns=['pop_index'] + [f'feature{i+1}' for i in range(n_features)]
    )
    
    for pop_idx in range(1, n_pop + 1):
        start_idx = n_points * (pop_idx - 1)
        end_idx = n_points * pop_idx
        population.iloc[start_idx:end_idx, 0] = pop_idx
        population.iloc[start_idx:end_idx, 1:(n_features + 1)] = create_random_n_k_lhs(n_points, n_features)
    
    return population


def get_pop_fitness(n_pop, population, p, distance_method):
    
    fitness = []
    
    for pop_idx in range(1, n_pop + 1):
        LHS_i = population[population['pop_index'] == pop_idx].drop(columns=['pop_index']).to_numpy()
        fitness_pop_i = compute_phip_lhs(LHS_i, p, distance_method)
        fitness.append(fitness_pop_i)
        
    fitness_df = pd.DataFrame({
        'pop_index': np.arange(1, n_pop + 1),
        'fitness': fitness
    })
    
    return fitness_df


def move_design(feature_j, LHS_i, best, point_is):
    
    for change_value in range(len(point_is)):
        
        point_i = point_is[change_value]
        
        if LHS_i[point_i, feature_j] != best[point_i, feature_j]:
            
            change_with = np.where(LHS_i[:, feature_j] == best[point_i, feature_j])[0][0]
            
            old_with = LHS_i[point_i, feature_j]
            LHS_i[point_i, feature_j] = best[point_i, feature_j]
            LHS_i[change_with, feature_j] = old_with
            
    return LHS_i


def rand_swap(LHS_i, feature_j):
    
    n_points = LHS_i.shape[0]
    
    swap_indexes = np.random.choice(np.arange(n_points), 2, replace=False)
    
    aux_value = LHS_i[swap_indexes[0], feature_j]
    LHS_i[swap_indexes[0], feature_j] = LHS_i[swap_indexes[1], feature_j]
    LHS_i[swap_indexes[1], feature_j] = aux_value
    
    return LHS_i


def add_fitness_to_pop(n_pop, population, fitness):
    
    updated_population = []
    
    for particle in range(1, n_pop + 1):
        
        population_particle = population[population['pop_index'] == particle]
        fitness_particle = fitness[fitness['pop_index'] == particle]
        population_particle['fitness'] = fitness_particle['fitness'].values[0]
        updated_population.append(population_particle)
    
    return pd.concat(updated_population)


def LaPSO_LHS(n_points, n_features, n_pop, max_iter, tol, distance_method, p, R):
    
    gbests_all = []
    
    population = get_population(n_points, n_features, n_pop)
    fitness = get_pop_fitness(n_pop, population, p, distance_method)
    
    gbest = population[population['pop_index'] == fitness.loc[fitness['fitness'].idxmax()]['pop_index']].drop(columns=['pop_index'])
    gbest['fitness'] = fitness.loc[fitness['fitness'].idxmax(), 'fitness']
    lbest = gbest.copy()
    
    gbests_all.append(gbest['fitness'].values[0])
    
    tol_iteration = 1000
    
    for iteration in range(1, max_iter + 1):
        
        new_population = pd.DataFrame(np.zeros((n_points * n_pop, n_features + 1)), columns=['pop_index'] + [f'feature{i+1}' for i in range(n_features)])
        
        if tol >= tol_iteration:
            break
        else:
            for particle in range(1, n_pop + 1):
                
                LHS_i = population[population['pop_index'] == particle].drop(columns=['pop_index']).to_numpy()
                
                for feature_j in range(n_features):
                    
                    how_much_p = np.random.randint(1, n_points + 1)
                    how_much_g = np.random.randint(1, n_points + 1)
                    
                    same_num_p = np.random.choice(np.arange(n_points), how_much_p, replace=False)
                    same_num_g = np.random.choice(np.arange(n_points), how_much_g, replace=False)
                    
                    LHS_i = move_design(feature_j, LHS_i, lbest.drop(columns=['fitness']).to_numpy(), same_num_p)
                    LHS_i = move_design(feature_j, LHS_i, gbest.drop(columns=['fitness']).to_numpy(), same_num_g)
                    
                    if np.random.uniform(0, 1) < R:
                        LHS_i = rand_swap(LHS_i, feature_j)
                    
                    start_idx = n_points * (particle - 1)
                    new_population.iloc[start_idx:start_idx + n_points, 0] = particle
                    new_population.iloc[start_idx:start_idx + n_points, 1:(n_features + 1)] = LHS_i
        
        new_fitness = get_pop_fitness(n_pop, new_population, p, distance_method)
        
        for particle in range(1, n_pop + 1):
            
            last_best = fitness[fitness['pop_index'] == particle]['fitness'].values[0]
            gbest_value = gbest['fitness'].values[0]
            particle_current = new_fitness[new_fitness['pop_index'] == particle]['fitness'].values[0]
            
            if particle_current > last_best:
                
                change_indexes = new_population['pop_index'] == particle
                population.loc[change_indexes, :] = new_population.loc[change_indexes, :]
                fitness.loc[particle - 1, 'fitness'] = new_fitness.loc[particle - 1, 'fitness']
                
                if particle_current > gbest_value:
                    
                    gbest = new_population[new_population['pop_index'] == particle].drop(columns=['pop_index'])
                    gbest['fitness'] = new_fitness.loc[new_fitness['fitness'].idxmax(), 'fitness']
                    gbests_all.append(gbest['fitness'].values[0])
                    tol = particle_current - gbest_value
        
        lbest = population[population['pop_index'] == fitness.loc[fitness['fitness'].idxmax(), 'pop_index']].drop(columns=['pop_index'])
        lbest['fitness'] = fitness.loc[fitness['fitness'].idxmax(), 'fitness']
    
    print(f"Results after {iteration} iterations")
    print(f"Last tolerance: {tol}")
    print("Best sampling with OLHS")
    
    return {"fitness": gbests_all, "last_LHS": gbest}




