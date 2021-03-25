#!/usr/bin/env python3
import os
import sys
import time
import torch
import numpy as np
from utils import save_model
from models import TensorTrain
from datasets import load_data

if __name__ == '__main__':
    dataset = sys.argv[1]
    mb_size = int(sys.argv[2])
    lr = float(sys.argv[3])
    K = int(sys.argv[4])
    n_epochs = int(sys.argv[5])
    run = int(sys.argv[6])
    result_file = sys.argv[7]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = 'TT_{}_{}_{}_{}_{}_{}'.format(
        dataset, mb_size, lr, K, n_epochs, run)

    data = load_data(dataset)
    data_train = torch.tensor(data.trn.x).to(device)
    data_val = torch.tensor(data.val.x).to(device)
    data_test = torch.tensor(data.tst.x).to(device)
    print(data_train.shape)

    start = time.time()
    Ks = [K]*(data_train.shape[1]+1)
    model = TensorTrain(Ks=Ks, device=device)
    model.hot_start(data_train, sub_sample_size=mb_size, n_starts=500)
    model.fit_model(data_train, mb_size=mb_size, n_epochs=n_epochs, lr=lr)
    end = time.time()
    print(end - start)

    nllh_train = model.svi.evaluate_loss(data_train) / len(data_train)
    nllh_val = model.svi.evaluate_loss(data_val) / len(data_val)
    nllh_test = model.svi.evaluate_loss(data_test) / len(data_test)

    print(nllh_train)
    print(nllh_val)
    print(nllh_test)

    save_model(model, model_name)

    output_string = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
        dataset, mb_size, lr, K, n_epochs, run,
        nllh_train, nllh_val, nllh_test)
    
    while not os.path.exists(result_file):
        time.sleep(1)
    
    # Simple semaphore lock
    os.rename(result_file, result_file + '_locked')

    file = open(result_file + '_locked', 'a')
    file.write(output_string)
    file.close()

    os.rename(result_file + '_locked', result_file)