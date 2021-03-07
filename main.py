# Imports
import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
from matplotlib.patches import Ellipse
from pyro.infer import SVI, TraceEnum_ELBO

import toy_data as toy_data
from models import gmm, CP

# Plot options
plt.style.use('seaborn-dark')
np.set_printoptions(precision=4)

# Run options
method = 'checkerboard'
N = 5000
K = 8

# Generate data
x = toy_data.inf_train_gen(method, batch_size=N)
X_tensor = torch.from_numpy(x).type(torch.float32)

# Instantiate model
#model = gmm(K=K)
model = CP(K=K)

#%% Train model

lr = 0.001
n_steps = 5000
show_likelihood = False


guide = model.guide
optimizer = pyro.optim.Adam({"lr": lr})
svi = SVI(model, guide, optimizer, TraceEnum_ELBO(num_particles=1))

pyro.clear_param_store()

for step in range(n_steps):
    model.train()
    loss = svi.step(X_tensor.float())
    if step % 500 == 0:
        model.eval()
        if show_likelihood:
            print('[iter {}]  loss: {:.4f}  Likelihood: {:.4f}'.format(step, loss, -model.get_likelihood(X_tensor).sum()))
        else:
            print('[iter {}]  loss: {:.4f}'.format(step, loss))


#%%

loc_est = model.loc.detach().numpy()
scale_est = model.scale.detach().numpy()
weight_est = model.weight.detach().numpy()
print(loc_est)
print(scale_est)
print(weight_est)

#%% Visualise

factor = 5
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ell = [Ellipse(xy=np.float64(loc_est[i, :]), width=scale_est[i, 0]*factor, height=scale_est[i, 1]*factor) for i in range(K)]

for i, e in enumerate(ell):
    ax.add_patch(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.3)
    e.set_facecolor(np.random.rand(3))

ax.scatter(x=x[:, 0], y=x[:, 1], color='k', alpha=0.3)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
plt.show()


#%% Generate data and plot alongside true data

# Samples
with torch.no_grad():
    x_samples = model().detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(x=x[:, 0], y=x[:, 1], color='k', alpha=0.3)
ax[0].set_title("True data")
ax[1].scatter(x=x_samples[:, 0], y=x_samples[:, 1], color='k', alpha=0.3)
ax[1].set_title("Generated data")

plt.show()


#%%
bins = 100
lim = [-5, 5, -5, 5]

# Points to compute LL
x = torch.linspace(lim[0], lim[1], bins)
y = torch.linspace(lim[2], lim[3], bins)
xx, yy = torch.meshgrid(x, y)

# Compute LL
vals = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), 1)
ll = np.exp(model.get_likelihood(data=vals))
ll = ll.reshape(bins, bins)

image = ll.squeeze().T
plt.imshow(image, interpolation='bicubic', cmap='winter', extent=np.array(lim), origin='lower')
plt.show()
#plt.savefig(data+'.png', format='png')

# Print total density. Should be close to 1
width = height = (lim[1]-lim[0])/bins
total_density = np.sum(ll)*width*height
print(f"Total density: {total_density:.3}")

#%%

# Compute density alternatively
density = model.get_density(data=vals)
density = density*width*height
print(f"Total density: {total_density:.3}")
