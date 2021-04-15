#!/usr/bin/env python3
import sys
import time
import torch
import argparse
import numpy as np
from utils import save_model
from models import TensorTrain
from datasets import load_data, all_datasets

def cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
        description='Script for training Tensor Train model')
    
    parser.add_argument(
        'model_name',
        help='Name to save model as.'
    )
    parser.add_argument(
        '--dataset',
        help='Dataset to fit model on.',
        choices=all_datasets,
        default='checkerboard'
    )
    parser.add_argument(
        '--K',
        help='Choice of K for tensor train model.',
        type=int,
        default=10
    )
    parser.add_argument(
        '--mb_size',
        help='Mini batch size.',
        type=int,
        default=256,
    )
    parser.add_argument(
        '--lr',
        help='Learning rate.',
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        '--epochs',
        help='Number of training epochs.',
        type=int,
        default=500,
    )
    parser.add_argument(
        '--subsample_size',
        help='Subsample training data.',
        type=int,
        default=None
    )
    parser.add_argument(
        '--optimal_order',
        help='Whether to order attributes (1 for True and 0 for False).',
        type=int,
        default=0
    )
    parser.add_argument(
        '--n_starts',
        help='Number of initializations',
        type=int,
        default=500
    )
    parser.add_argument(
        '--early_stopping',
        help='Whether to use early stopping (1 for True and 0 for False).',
        type=int,
        default=0
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = cli()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = load_data(
        args.dataset, 
        optimal_order=bool(args.optimal_order),
        subsample_size=args.subsample_size)
    
    data_train = torch.tensor(data.trn.x).to(device)
    data_val = torch.tensor(data.val.x).to(device)
    del data

    train_start = time.time()

    Ks = [args.K]*(data_train.shape[1]+1)
    model = TensorTrain(Ks=Ks, device=device)
    model.hot_start(
        data_train, subsample_size=args.subsample_size, n_starts=args.n_starts)
    model.fit_model(
        data_train, data_val=data_val, mb_size=args.mb_size,
        n_epochs=args.epochs, lr=args.lr,
        early_stopping=bool(args.early_stopping))

    train_end = time.time()
    print('Training time: {:.1f} seconds'.format(train_end-train_start))
    
    save_model(model, args.model_name)