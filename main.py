# Imports
import pyro
import torch
import numpy as np
from datasets import load_data
import matplotlib.pyplot as plt
from models import GaussianMixtureModel, CPModel, TensorTrain
from utils import plot_density, plot_density_alt, plot_train_loss
pyro.enable_validation(True)

# Plot options
plt.style.use('seaborn-dark')
np.set_printoptions(precision=3)

# Select dataset and model
data_type = 'checkerboard'
model_type = 'TensorTrain'
K = [2,2,2]

# Load data
data = load_data(data_type)
data_train = torch.tensor(data.trn.x)
data_val = torch.tensor(data.val.x)
data_test = torch.tensor(data.tst.x)

#data = torch.column_stack((data, torch.zeros(5000))) # Test for 3D ---- add 0s as third element
# Instantiate model
model = eval(model_type)(K)

#%%
dens = model.unit_test_multidimensional([-5, 5, -5, 5], n_points=400)
print(f"Total density: {dens.item()}")


#%% Train the model
if model_type == 'TensorTrain':
    print("[Initialising tensor train parameters]")
    model.hot_start(data_train, n_starts=250)
    print("[Tensor train initialised]")
model.fit_model(data_train, n_steps=250)

#%%
# Log likelihood of data
print(f'Negative log likelihood of data: {-model.log_likelihood(data_train).item():.4f}')

#%%
# Plot train loss
plot_train_loss(model)

#%%
# Plot density
#plot_density(model, data)
plot_density_alt(model, data_train)

#%%
#model.unit_test([(-5, 5), (-5, 5)], opts={'epsabs': 0.01}) # Allow a larger error for faster computation

total_density = model.unit_test_alt(3000)
print(f"Total density: {total_density:.4}")

total_density = model.unit_test_multidimensional([-5, 5, -5, 5], 3000)
print(f"Total density: {total_density:.4}")


#%%
# Sample data
with torch.no_grad():
    sampled_data = model().detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(x=data_train[:, 0], y=data_train[:, 1], color='k', alpha=0.3)
ax[0].set_title("True data")
ax[1].scatter(x=sampled_data[:, 0], y=sampled_data[:, 1], color='k', alpha=0.3)
ax[1].set_title("Generated data")

plt.show()

