#!/usr/bin/env python3
import os
import pyro
import torch
import numpy as np
import pandas as pd
from utils import load_model
from datasets import load_data

model_list = os.listdir('models/')
result_file = 'results/tt_subsample_results.txt'
col_names = [
    'dataset', 'K', 'mb_size', 'lr', 'epochs', 'subsample_size',
    'optimal_order', 'n_start', 'early_stop', 'run', 'nllh_train', 'nllh_val', 'nllh_test']

results = {key: list() for key in col_names}

for model_name in model_list:
    dataset, K, mb_size, lr, epochs, subsample_size, order, n_start, early_stop, run = model_name[:-3].split('_')[1:]

    model = load_model(model_name)
   
    data = load_data(
        dataset,
        subsample_size=int(subsample_size),
        optimal_order=bool(int(order)))
    
    data_train = torch.tensor(data.trn.x)
    data_val = torch.tensor(data.val.x)
    data_test = torch.tensor(data.tst.x)
    del data
    
    model.eval()
    nllh_train = model.nllh(data_train) / len(data_train)
    nllh_val = model.nllh(data_val) / len(data_val)
    nllh_test = model.nllh(data_test) / len(data_test)
    
    results['dataset'].append(dataset)
    results['K'].append(K)
    results['mb_size'].append(mb_size)
    results['lr'].append(lr)
    results['epochs'].append(epochs)
    results['subsample_size'].append(subsample_size)
    results['optimal_order'].append(order)
    results['n_start'].append(n_start)
    results['early_stop'].append(early_stop)
    results['run'].append(run),
    results['nllh_train'].append(nllh_train),
    results['nllh_val'].append(nllh_val)
    results['nllh_test'].append(nllh_test)

df = pd.DataFrame.from_dict(results)
df.to_csv(result_file)