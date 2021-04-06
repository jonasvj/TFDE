#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

col_names = ['dataset', 'mb_size', 'lr', 'K', 'n_epochs', 'run',
    'nllh_train', 'nllh_val', 'nllh_test']

df = pd.read_csv(
    'results/tt_results_final.txt', sep='\t', header=None,
    names=col_names)

groups = df.groupby(['dataset', 'mb_size', 'lr', 'n_epochs'])

for group, frame in groups:
    name = 'TT_{}_{}_{}_{}'.format(*group)
    mean = frame.groupby('K').mean()
    sem = frame.groupby('K').sem()
    min_ = frame.groupby('K').min()
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
    