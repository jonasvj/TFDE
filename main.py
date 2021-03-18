# Imports
import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
import toy_data as toy_data
from models import GaussianMixtureModel, CPModel, TensorTrain
from utils import plot_density, plot_density_alt, plot_train_loss
pyro.enable_validation(True)

# Plot options
plt.style.use('seaborn-dark')
np.set_printoptions(precision=3)

# Select dataset and model
method = 'checkerboard'
model_type = 'TensorTrain'
K = [2,2,2]

# Generate data
data = torch.tensor(toy_data.inf_train_gen(method, batch_size=5000), dtype=torch.float)
#data = torch.column_stack((data, torch.zeros(5000))) # Test for 3D ---- add 0s as third element
# Instantiate model
model = eval(model_type)(K)

#%%
dens = model.unit_test_multidimensional([-5, 5, -5, 5], n_points=400)
print(f"Total density: {dens.item()}")


#%% Train the model
if model_type == 'TensorTrain':
    print("[Initialising tensor train parameters]")
    model.hot_start(data, n_starts=250)
    print("[Tensor train initialised]")
model.fit_model(data, n_steps=2000)

#%%
# Log likelihood of data
print(f'Log likelihood of data: {-model.log_likelihood(data).item():.4f}')

#%%
# Plot train loss
plot_train_loss(model)

#%%
# Plot density
#plot_density(model, data)
plot_density_alt(model, data)

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
ax[0].scatter(x=data[:, 0], y=data[:, 1], color='k', alpha=0.3)
ax[0].set_title("True data")
ax[1].scatter(x=sampled_data[:, 0], y=sampled_data[:, 1], color='k', alpha=0.3)
ax[1].set_title("Generated data")

plt.show()

