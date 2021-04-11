#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def tt_params(K, M):
    return K + 3*M*K**2

def bic(n, k, nllh_per_sample):
    log_lh = -1*nllh_per_sample*n
    return k*np.log(n) - 2*log_lh

def aic(n, k, nllh_per_sample):
    log_lh = -1*nllh_per_sample*n
    return 2*k - 2*log_lh

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


col_names = ['dataset', 'mb_size', 'lr', 'K', 'n_epochs', 'run',
    'nllh_train', 'nllh_val', 'nllh_test']

df = pd.read_csv(
    'results/tt_results_final.txt', sep='\t', header=None,
    names=col_names)

# Add new columns
df['n_train'] = df.apply(lambda row: sizes[row.dataset]['n_train'], axis=1)
df['n_val'] = df.apply(lambda row: sizes[row.dataset]['n_val'], axis=1)
df['n_test'] = df.apply(lambda row: sizes[row.dataset]['n_test'], axis=1)
df['M'] = df.apply(lambda row: sizes[row.dataset]['M'], axis=1)
df['n_params'] = df.apply(lambda row: tt_params(row.K, row.M), axis=1)

df['bic_train'] = df.apply(lambda row: bic(row.n_train, row.n_params, row.nllh_train), axis=1)
df['bic_val'] = df.apply(lambda row: bic(row.n_val, row.n_params, row.nllh_val), axis=1)
df['bic_test'] = df.apply(lambda row: bic(row.n_test, row.n_params, row.nllh_test), axis=1)

df['aic_train'] = df.apply(lambda row: aic(row.n_train, row.n_params, row.nllh_train), axis=1)
df['aic_val'] = df.apply(lambda row: aic(row.n_val, row.n_params, row.nllh_val), axis=1)
df['aic_test'] = df.apply(lambda row: aic(row.n_test, row.n_params, row.nllh_test), axis=1)

groups = df.groupby(['dataset', 'mb_size', 'lr', 'n_epochs'])

for group, frame in groups:
    name = 'TT_{}_{}_{}_{}'.format(*group)
    mean = frame.groupby('K').mean()
    sem = frame.groupby('K').sem()
    min_ = frame.groupby('K').min()

    # NLLH per sample vs K
    opt_K_avg = mean.nllh_val.idxmin()
    opt_K_min = min_.nllh_val.idxmin()
    fig, ax = plt.subplots(figsize=(8,6))
    ax.errorbar(mean.index, mean.nllh_train, yerr=sem.nllh_train, fmt='.:',
        label='Train', alpha=.75, capsize=3, capthick=1)
    ax.errorbar(mean.index, mean.nllh_val, yerr=sem.nllh_val, fmt='.:',
        label='Validation', alpha=.75, capsize=3, capthick=1)
    ax.set_xlabel('K')
    ax.set_ylabel('NLLH per sample')
    ax.set_title(name + f', Opt K(avg): {opt_K_avg}, Opt K(min): {opt_K_min}')
    ax.legend()
    fig.savefig('plots/' + name + '_loss_vs_K.pdf')
    plt.close()

    # BIC vs K
    opt_K_avg = mean.bic_val.idxmin()
    opt_K_min = min_.bic_val.idxmin()
    fig, ax = plt.subplots(figsize=(8,6))
    ax.errorbar(mean.index, mean.bic_train, yerr=sem.bic_train, fmt='.:',
        label='Train', alpha=.75, capsize=3, capthick=1)
    ax.errorbar(mean.index, mean.bic_val, yerr=sem.bic_val, fmt='.:',
        label='Validation', alpha=.75, capsize=3, capthick=1)
    ax.set_xlabel('K')
    ax.set_ylabel('BIC')
    ax.set_title(name + f', Opt K(avg): {opt_K_avg}, Opt K(min): {opt_K_min}')
    ax.legend()
    fig.savefig('plots/' + name + '_BIC_vs_K.pdf')
    plt.close()

    # AIC vs K
    opt_K_avg = mean.aic_val.idxmin()
    opt_K_min = min_.aic_val.idxmin()
    fig, ax = plt.subplots(figsize=(8,6))
    ax.errorbar(mean.index, mean.aic_train, yerr=sem.aic_train, fmt='.:',
        label='Train', alpha=.75, capsize=3, capthick=1)
    ax.errorbar(mean.index, mean.aic_val, yerr=sem.aic_val, fmt='.:',
        label='Validation', alpha=.75, capsize=3, capthick=1)
    ax.set_xlabel('K')
    ax.set_ylabel('AIC')
    ax.set_title(name + f', Opt K(avg): {opt_K_avg}, Opt K(min): {opt_K_min}')
    ax.legend()
    fig.savefig('plots/' + name + '_AIC_vs_K.pdf')
    plt.close()
