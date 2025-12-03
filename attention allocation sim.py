#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 07:02:02 2025

@author: ntsikamajozi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Tuple, List
import tqdm
import random
import copy

# -------------------------------
# Model
# -------------------------------

def platform_reward(exposure:float, baseline:float, A:float, gamma:float) -> float:
    
    '''
    platform reward function

    Parameters
    ----------
    exposure : float
        popularity (>=0).
    baseline : float
        minimum reward.
    A : float
        virality strength multiplier (scaling).
    gamma : float
        virality exponent (gamma > 1 amplifies rich-get-richer).

    Returns
    -------
    float
        scalar reward value awarded by platform.

    '''
    
    return baseline + A*(exposure)

def content_exposure(score:float, total_pool:float) -> float:
    if total_pool <= 0:
        return 0.0
    return score/total_pool

# -------------------------------
# Creator definition
# -------------------------------

class Creator:
    
    def __init__(self, honesty: float):
        
        '''
        Attributes
        
        - honesty in [0,1]. 1 = fully authentic, 0 = fully optimised.
        - last reward
        -----------------------------
        Returns
        
        None.
        '''
        self.honesty = float(np.clip(honesty,0.0,1.0))
        self.last_reward = 0.0
        
    def calc_score(self, params: Dict[str, Any]) -> float:
        
        '''
        map honesty to content score (intrinsic value)
        simple linear parametric mix: authenticity might increase artistic depth, but optimised content may
        have short-term clickbait effect. Adjust to experiment. 
        score = a*honesty + b*(1-honesty)
        We also add in small noise for realism
        '''
        
        a = params.get('alpha_authentic', 1.0)
        b = params.get('alpha_optimised', 0.6)
        noise_std = params.get('production_noise', 0.01)
        base = a*self.honesty + b*(1-self.honesty)
        return max(0.0, base + np.random.normal(0,noise_std))
        
# -------------------------------
# Population definition
# -------------------------------

class Population:
    
    def __init__(self, creators: List[Creator]):
        
        self.creators = creators
        
    @classmethod
    def random_h(cls, n:int, honest_dist = 'uniform', **kwargs):

        if honest_dist == 'uniform':
            hs = np.random.rand(n)
        elif honest_dist == 'bimodal':
            hs = np.clip(np.concatenate([np.random.beta(5,2, size = n//2), np.random.beta(2,5, size = n - n//2)]), 0, 1)
        elif honest_dist == 'normal':
            hs = np.clip(np.random.normal(0.5,0.15, size=n), 0, 1)
        elif honest_dist == 'beta':
            a = kwargs.get('a', 8)
            b = kwargs.get('b', 2)
            hs = np.clip(np.random.beta(a = a, b = b, size = n), 0, 1)
        else:
            raise ValueError("unknown honesty distribution")
            
        creators = [Creator(float(h)) for h in hs]
        return cls(creators)
    
    def honesty_array(self) -> np.ndarray:
        return np.array([c.honesty for c in self.creators])
        
    def copy(self):
        return Population([copy.copy(c) for c in self.creators])
# -------------------------------
# Agent-based simulation
# -------------------------------

def run_agent_based(
    pop: Population,
    platform_params: Dict[str, Any],
    production_params: Dict[str, Any],
    steps: int = 200,
    imitation_noise: float = 0.05,
    seed: int = None
) -> Dict[str,Any]:
    
    '''
    agent-based simulation
    
    - at each step, players produce a content score
    - platform computes exposures and reward using virality function
    - creators update their honesty by imitating others proportional to reward (pairewise sampling)
    -----------------------------
    Returns
    
    - Dict with time series of mean honesty, distributed snapshots, final honesty
    '''

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    n = len(pop.creators)
    mean_honesty = np.zeros(steps)
    honesty_trace = np.zeros((steps, n))
    for t in range(steps):
        
        baseline = platform_params.get('baseline', 0.1)
        A = platform_params.get('A', 1.0)
        gamma = platform_params.get('gamma', 1.0)
        
        scores = np.array([c.calc_score(production_params) for c in pop.creators])
        total_scores =  (scores**gamma).sum()
        exposures = np.array([content_exposure(s**gamma, total_scores) for s in scores])
        
        rewards = platform_reward(exposures, baseline, A, gamma)
        
        for i,c in enumerate(pop.creators):
            
            c.last_reward = rewards[i]
        
        beta = platform_params.get('selection_strength', 5.0)
        for i,c in enumerate(pop.creators):
            j = np.random.randint(0, n)
            
            payoff_i = rewards[i]
            payoff_j = rewards[j]
            prob_adopt = 1/(1 + np.exp(-beta * (payoff_j - payoff_i)))    
            
            if np.random.rand() < prob_adopt:
                
                new_honesty = pop.creators[j].honesty + np.random.normal(0, imitation_noise)
                pop.creators[i].honesty = float(np.clip(new_honesty,0.0, 1.0))
        
        mean_honesty[t] = pop.honesty_array().mean()
        honesty_trace[t, :] = pop.honesty_array()
        
    return {
        'mean_honesty': mean_honesty,
        'honesty_trace': honesty_trace,
        'final_honesty': pop.honesty_array()
    }

# -------------------------------
# repicator dynamics simulation (discretised types)
# -------------------------------

def discrete_honesty(pop: Population, intervals: int = 41):
    
    honesty_vals = pop.honesty_array()
    types = np.linspace(0, 1, intervals)
    
    midpoints = (types[:-1] + types[1:])/2
    bin_edges = np.concatenate(([0.0], midpoints, [1.000001]))
    
    idx = np.digitize(honesty_vals, bin_edges) - 1
    freq = np.bincount(idx, minlength = len(types)).astype(float)
    
    freq /= freq.sum()
    return types, freq

def run_replicator(
    honesty_types: np.ndarray,
    freq: np.ndarray,
    platform_params: Dict[str, Any],
    production_params: Dict[str, Any],
    dt: float = 0.1,
    steps: int = 1000
) -> Dict[str, Any]:
    
    '''
    discrete replicator dynamics
    
    - freq: proportion vector of population over honesty_types (add up to 1)
    - Fitness of type i is expected rewards under platform model given other types' population
    - approximate exposure with type*mean_score_of_type/total_scores
    -----------------------------
    Returns
    
    - Dict with types, history of frequencies, final_freq
    '''
    
    types = honesty_types
    freq = freq.astype(float)
    history = np.zeros((steps,len(types)))
    
    baseline = platform_params.get('baseline', 0.1)
    A = platform_params.get('A', 1.0)
    gamma = platform_params.get('gamma', 1.0)
    
    type_scores = np.array([(production_params.get('alpha_authentic', 1.0)*h 
                             + production_params.get('alpha_optimised', 0.6)*(1-h)) 
                            for h in types])
    
    for t in range(steps):
        
        total_pool_score = (freq * type_scores**gamma).sum()
        
        exposures = np.array([content_exposure(score**gamma, total_pool_score) for score in type_scores])
        rewards = platform_reward(exposures, baseline, A, gamma)
        
        avg_reward = (freq * rewards).sum()
        df = freq * (rewards - avg_reward)
        freq = freq + dt * df
        
        freq = np.clip(freq, 1e-10, None)
        freq /= freq.sum()
        history[t, :] = freq
    
    return{
        'types': types,
        'history': history,
        'final_freq': freq
    }
    
# -------------------------------
# utilities: experiment runners and plots
# -------------------------------
    
def make_phase_diagram_replicator(
    honesty_grid: np.linspace,
    A_vals: List[float],
    gamma_vals: List[float],
    platform_base: float,  
    production_params: Dict[str, Any],
    steps: int = 1000    
) -> pd.DataFrame:

    '''
    interate over A and gamma and record final_mean_honesty of each pair
    -----------------------------
    Returns
    
    - Dataframe with columns: A, gamma, mean_honestly_final
    '''
    
    rows = []
    
    types = np.linspace(0,1.0,21)
    
    init_freq = np.ones_like(types)/len(types)
    
    for A in tqdm.tqdm(A_vals, desc = 'A sweep'):
        for gamma in gamma_vals:
            
            platform = {'baseline': platform_base, 'A':A, 'gamma':gamma}
            out = run_replicator(types, init_freq.copy(), platform, production_params, dt=0.05, steps=steps)
            final_freq = out['final_freq']
            mean_honesty = (out['types']*final_freq).sum()
            rows.append({'A':A, 'gamma':gamma, 'mean_honesty':mean_honesty})
    
    return pd.DataFrame(rows)

def plot_time_series(result: Dict[str, Any], title = 'Mean Honesty Over Time'):
    
    '''
    plot mean honesty against time step
    '''
    
    plt.figure(figsize =(6,4))
    t = np.arange(len(result['mean_honesty']))
    plt.plot(t, result['mean_honesty'], linewidth = 2)
    plt.ylim(0,1)
    plt.xlabel('Time step')
    plt.ylabel('Mean honesty')
    plt.title(title)
    plt.grid(alpha = 0.25)
    plt.tight_layout()
    plt.show()
    
def plot_phase_heatmap(df: pd.DataFrame, A_vals: List[float], gamma_vals: List[float]):
    
    '''
    plot heat map of A, gamma vs. final mean honesty
    '''
    
    pivot = df.pivot(index = 'gamma', columns = 'A', values = 'mean_honesty')
    plt.figure(figsize = (6,5))
    plt.imshow(pivot.values, origin = 'lower', aspect = 'auto', extent = [min(A_vals), max(A_vals), min(gamma_vals), max(gamma_vals)])
    plt.colorbar(label = 'Final Mean Honesty')
    plt.xlabel('total available attention')
    plt.ylabel('$\gamma$ (virality exponent)')
    plt.title('Phase Diagram: Final Mean Honesty')
    plt.tight_layout()
    plt.show()
    
# -------------------------------
# experiment (main)
# -------------------------------
    
if __name__ == '__main__':
    
    N = 2000
    steps = 300
    platform_params = {
        'baseline': 0.01,
        'A': 1.0,
        'gamma': 20,
        'selection_strength': 50.0
    }
    production_params = {
        'alpha_authentic': 0.6,
        'alpha_optimised': 1.0,
        'production_noise': 0.02
        
    }
    save_results = False
    
    pop = Population.random_h(N, honest_dist = 'beta', a = 8, b = 2)
    pop_original = pop.copy()
    
    # agent-based simulation
    agent_result = run_agent_based(pop, platform_params, production_params, steps = steps, imitation_noise = 0.02, seed = 42)
    plot_time_series(agent_result, title = 'Agent-Based: Mean Honesty')
    
    # replicator simulation
    types, init_freq = discrete_honesty(pop_original)
    
    replicator_result = run_replicator(types, init_freq, platform_params, production_params, dt = 0.05, steps = steps)
    
    mean_over_time = (replicator_result['history'] * replicator_result['types']).sum(axis = 1)
    plt.figure(figsize = (6,4))
    plt.plot(mean_over_time)
    plt.ylim(0,1)
    plt.xlabel('Time step')
    plt.ylabel('Mean honesty')
    plt.title('Replicator: Mean Honesty')    
    plt.grid(alpha = 0.25)
    plt.tight_layout()
    plt.show()
    
    A_vals = np.linspace(1,5,51)
    gamma_vals = np.linspace(1,5,51)
    df = make_phase_diagram_replicator(types, A_vals, gamma_vals, platform_base = 0.01, production_params = production_params, steps = 10)
    plot_phase_heatmap(df, A_vals, gamma_vals)
    if save_results == True:
        df.to_csv('phase_diagram_results.csv', index = False)
        print('Simylation complete! Results saved to phase_diagram_results.csv')
        
    
    
    
    
    
