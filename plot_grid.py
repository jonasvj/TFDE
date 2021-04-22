#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def tt_params(K, M):
    return K + 3*M*K**2

def tt_dof(K, M):
    return (K-1) + M*K*(K-1) + 2*M*K**2

def bic(n, k, nllh_per_sample):
    log_lh = -1*nllh_per_sample*n
    return k*np.log(n) - 2*log_lh

def aic(n, k, nllh_per_sample):
    log_lh = -1*nllh_per_sample*n
    return 2*k - 2*log_lh

def n_params(model, K, M):
    if model == 'TT':
        return (K-1) + M*K*(K-1) + 2*M*K*K
    elif model == 'CP':
        return (K-1) + 2*M*K
    elif model == 'GMM':
        return (K-1) + (2*M + M*(M-1)/2)*K


sizes = {
    'power': {'n_train': 1659917, 'n_val': 184435, 'n_test': 204928, 'M': 6}, 
    'gas': {'n_train': 852174, 'n_val': 94685, 'n_test': 105206, 'M': 8},
    'hepmass': {'n_train': 315123, 'n_val': 35013, 'n_test': 174987, 'M': 21},
    'miniboone': {'n_train': 29556, 'n_val': 3284, 'n_test': 3648, 'M': 43},
    'bsds300': {'n_train': 1000000, 'n_val': 50000, 'n_test': 250000, 'M': 63},
    '8gaussians': {'n_train': 30000, 'n_val': 30000, 'n_test': 30000, 'M': 2},
    'checkerboard': {'n_train': 30000, 'n_val': 30000, 'n_test': 30000, 'M': 2},
    '2spirals': {'n_train': 30000, 'n_val': 30000, 'n_test': 30000, 'M': 2}
    }

df = pd.read_csv('results/grid_results.txt', index_col=0)
df_gmm = pd.read_csv('results/gmm_results.txt', index_col=0)
df = df.append(df_gmm, ignore_index=True)

df = df[df.optimal_order == 1]
print(df)
# Add new columns
df['M'] = df.apply(lambda row: sizes[row.dataset]['M'], axis=1)
df['dof'] = df.apply(lambda row: n_params(row.model_type, row.K, row.M), axis=1)
datasets = ['hepmass', 'miniboone']
subsample_sizes = [1750, 7000, 28000]

groups = df.groupby(['dataset', 'subsample_size'])

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 12),
    sharex='all', sharey='row')

for i, (group, frame) in enumerate(groups):
    row_idx = datasets.index(group[0])
    col_idx = subsample_sizes.index(group[1])
    model_groups = frame.groupby(['model_type'])

    for model, model_frame in model_groups:
        mean = model_frame.groupby('dof').mean()
        sem = model_frame.groupby('dof').sem()
        min_ = model_frame.groupby('dof').min()

        axes[row_idx, col_idx].errorbar(
            mean.index, mean.nllh_test, yerr=sem.nllh_test, fmt='.:',
            label=model, alpha=.75, capsize=3, capthick=1)
    
    axes[row_idx, col_idx].set_xlabel('Free parameters')
    axes[row_idx, col_idx].set_ylabel(f'Test NLLH per sample ({group[0]})')
    axes[row_idx, col_idx].set_title(f'Subsample size: {group[1]}')
    axes[row_idx, col_idx].legend()
  

fig.savefig('plots/' + 'grid_plot.pdf')
plt.close()