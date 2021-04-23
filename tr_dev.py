#%%
# Imports
import pyro
import sys
import torch
import numpy as np
from datasets import load_data
import matplotlib.pyplot as plt
from models import GaussianMixtureModel, CPModel, TensorTrain, GaussianMixtureModelFull, TensorRing
from utils import plot_density, plot_density_alt, plot_train_loss
pyro.enable_validation(True)

import os
#os.chdir('TFDE')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(f'Using device: {device}')

device = 'cpu'

#print(torch.cuda.device_count())
#sys.exit(1)
# Plot options
plt.style.use('seaborn-dark')
np.set_printoptions(precision=3)

# Select dataset and model
data_type = 'checkerboard'
model_type = 'GaussianMixtureModelFull'
n_dim = 2 # dimensions of data
K = [8]*(n_dim + 1) # 35 for full GMM = TT with Ks = 6


# Load data
data = load_data(data_type, optimal_order=False)

data_train = torch.tensor(data.trn.x).to(device)
data_val = torch.tensor(data.val.x).to(device)
data_test = torch.tensor(data.tst.x).to(device)

#data_train = data_train[:1000,:]
#data = torch.column_stack((data, torch.zeros(5000))) # Test for 3D ---- add 0s as third element
# Instantiate model
model = eval(model_type)(K, device=device)
#model.init_from_data(data_train, k_means=True)

#%%

#dens = model.unit_test_multidimensional(np.array([-5, 5]*n_dim).reshape(-1), n_points=400)
#print(f"Total density: {dens.item()}")


#%% Train the model
if model_type in ['TensorTrain', 'TensorRing'] :
    print("[Initialising tensor train/ring parameters]")
    model.hot_start(data_train, n_starts=250)
    print("[Tensor train/ring initialised]")
else:
    model.init_from_data(data_train, k_means=True)
model.fit_model(data_train, mb_size=len(data_train), lr=3e-4, n_epochs=2000)


#sys.exit(1)

#%%
# Log likelihood of data
model.eval()
print(f'Negative log likelihood of data: {-model.log_likelihood(data_train).item()/len(data_train):.4f}')

#%%
# Plot train loss
#plot_train_loss(model)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 6))
plot_train_loss(model, axes[0])
plot_density(model, data_train.detach().cpu().numpy(), 
			density_grid=[-3, 3, -3, 3],
			axes=axes[1:])

#plt.show()
plt.savefig("tmp.png")
