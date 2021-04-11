#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
from utils import plot_train_loss, load_model

model_list = os.listdir('models/')

for model_name in model_list:
    model = load_model(model_name)
    fig, ax = plt.subplots(figsize=(8,6))
    plot_train_loss(model, ax=ax)
    fig.savefig('plots/' + model_name.split('.pt')[0] + '_learning_curve.pdf')
    plt.close()