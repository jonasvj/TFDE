#!/usr/bin/env python3
import os
import numpy as np
from datasets import root
from ffjord.lib.toy_data import inf_train_gen

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    rng = np.random.RandomState(seed=seed)
    batch_size = 15000

    # Generate data sets
    data_8g = inf_train_gen('8gaussians', rng=rng, batch_size=batch_size)
    data_cb = inf_train_gen('checkerboard', rng=rng, batch_size=batch_size)
    data_2s = inf_train_gen('2spirals', rng=rng, batch_size=batch_size)

    # Save datasets
    if not os.path.isdir(root + 'synthetic'):
        os.mkdir(root + 'synthetic')

    np.save(root + 'synthetic/8gaussians.npy', data_8g)
    np.save(root + 'synthetic/checkerboard.npy', data_cb)
    np.save(root + 'synthetic/2spirals.npy', data_2s)