#!/usr/bin/env python3
import os
import pyro
import torch
from utils import load_model
from datasets import load_data
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete

model_list = os.listdir('models/')
tt_result_file = open('results/tt_results_final.txt', 'r')

models_w_results = list()
for line in tt_result_file:
    dataset, mb_size, lr, K, n_epochs, run = line.split()[:6]
    models_w_results.append('TT_{}_{}_{}_{}_{}_{}.pt'.format(
        dataset, mb_size, lr, K, n_epochs, run))
tt_result_file.close()

models_wo_results = sorted(set(model_list) - set(models_w_results))

tt_result_file = open('results/tt_results_final.txt', 'a')

for model_name in models_wo_results:
    model_type, dataset, mb_size, lr, K, n_epochs, run = model_name.strip()[:-3].split('_')
    if dataset == 'bsds300':
        continue
    model = load_model(model_name)
    adam = pyro.optim.Adam({"lr": float(lr)})
    svi = SVI(model, model.guide, adam, loss=TraceEnum_ELBO())

    data = load_data(dataset)
    data_train = torch.tensor(data.trn.x)
    data_val = torch.tensor(data.val.x)
    data_test = torch.tensor(data.tst.x)
    del data

    model.eval()
    with torch.no_grad():
        nllh_train = svi.evaluate_loss(data_train) / len(data_train)
        nllh_val = svi.evaluate_loss(data_val) / len(data_val)
        nllh_test = svi.evaluate_loss(data_test) / len(data_test)
    
    output_string = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
        dataset, mb_size, lr, K, n_epochs, run,
        nllh_train, nllh_val, nllh_test)
    
    tt_result_file.write(output_string)

tt_result_file.close()