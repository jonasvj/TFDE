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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Plot options
plt.style.use('seaborn-dark')
np.set_printoptions(precision=3)

# Select dataset and model
data_type = 'checkerboard'
model_type = 'TensorRing'
n_dim = 2 # dimensions of data
K = [6]*(n_dim + 1)

# Load data
data = load_data(data_type, optimal_order=False)
data_train = torch.tensor(data.trn.x).to(device)
data_val = torch.tensor(data.val.x).to(device)
data_test = torch.tensor(data.tst.x).to(device)

#data = torch.column_stack((data, torch.zeros(5000))) # Test for 3D ---- add 0s as third element
# Instantiate model
model = eval(model_type)(K, device=device)


#%%
#dens = model.unit_test_multidimensional(np.array([-5, 5]*n_dim).reshape(-1), n_points=10)
#print(f"Total density: {dens.item()}")


#%% Train the model
if model_type in ['TensorTrain', 'TensorRing'] :
    print("[Initialising tensor train/ring parameters]")
    model.hot_start(data_train, n_starts=250)
    print("[Tensor train/ring initialised]")
else:
    model.init_from_data(data_train, k_means=True)
model.fit_model(data_train, mb_size=len(data_train), n_epochs=20000)

#%%
# Log likelihood of data
model.eval()
print(f'Negative log likelihood of data: {-model.log_likelihood(data_train).item()/len(data_train):.4f}')

#%%
# Plot train loss
#plot_train_loss(model)
plot_density_alt(model, data_train.detach().cpu().numpy())