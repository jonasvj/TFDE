# Imports
import numpy as np
import pyro
import torch
from torch.distributions import constraints
from torch import nn
import pyro.distributions as dist
import matplotlib.pyplot as plt
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.ops.indexing import Vindex
from matplotlib.patches import Ellipse
import toy_data as toy_data
import seaborn as sns

plt.style.use('seaborn-dark')

use_ffjord = True
data = '2spirals'
N = 1000
K = 8


# %%
# Generate data
if not use_ffjord:
    N = 2500
    K = 3
    means = np.random.rand(K, 2)*2 + np.random.randint(-10,10)
    scales = np.random.rand(K, 2)**2
    x = np.zeros(N*K)
    y = np.zeros(N*K)
    for i in range(K):
        x[(i*N):((i+1)*N)] = np.random.normal(loc=means[i, 0], scale=scales[i, 0], size=N)
        y[(i * N):((i + 1) * N)] = np.random.normal(loc=means[i, 1], scale=scales[i, 1], size=N)
        plt.scatter(x=x[(i*N):((i+1)*N)], y=y[(i*N):((i+1)*N)], alpha=0.4)
    plt.show()

    X = np.vstack((x, y)).T
    np.random.shuffle(X)
    X_tensor = torch.from_numpy(X)
else:
    x = toy_data.inf_train_gen(data,batch_size=N)
    plt.scatter(x=x[:, 0], y=x[:, 1], alpha=0.4, color='k')
    plt.show()
    X_tensor = torch.from_numpy(x).type(torch.float32)

#%% Model definition

class gmm(nn.Module):

    def __init__(self, K):
        super(gmm, self).__init__()
        self.K = K
        self.loc = torch.ones((self.K, 2))
        self.scale = torch.ones((self.K, 2))
        self.weight = torch.ones(self.K)

    @config_enumerate
    def model(self, data=None):
        N = len(data) if data is not None else 1000
        loc = pyro.param("loc", torch.rand((self.K, 2)))
        scale = pyro.param("scale", torch.ones((self.K, 2)), constraints.positive)
        weight = pyro.param("weight", torch.ones(self.K), constraints.simplex)

        with pyro.plate("data", N):
            assignment = pyro.sample('assignment', dist.Categorical(weight)).long()
            fn_dist = dist.Normal(Vindex(loc)[..., assignment, :], Vindex(scale)[..., assignment, :]).to_event(1)

            x_samples = pyro.sample("x_samples",
                                    fn_dist, obs=data)

        self.loc = loc
        self.scale = scale
        self.weight = weight
        return x_samples

    def guide(self, data):
        pass

model = gmm(K=K)

# %%
lr = 3e-4
n_steps = 20000


guide = model.guide
optimizer = pyro.optim.Adam({"lr": lr})
svi = SVI(model.model, guide, optimizer, TraceEnum_ELBO(num_particles=1, max_plate_nesting=1))

pyro.clear_param_store()

for step in range(n_steps):
    model.train()
    loss = svi.step(X_tensor.float())
    if step % 2000 == 0:
        print('[iter {}]  loss: {:.4f}'.format(step, loss))
        #print(pyro.param("weight"))


print(pyro.param("loc"))
print(pyro.param("scale"))
print(pyro.param("weight"))


#%% Visualise

K = model.K
loc_est = model.loc.detach().numpy()
scale_est = model.scale.detach().numpy()


factor = 5
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ell = [Ellipse(xy=np.float64(loc_est[i, :]), width=scale_est[i, 0]*factor, height=scale_est[i, 1]*factor) for i in range(K)]

for e in ell:
    ax.add_patch(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.7)
    e.set_facecolor(np.random.rand(3))

ax.scatter(x=x[:,0], y=x[:,1], color='k', alpha=0.3)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
plt.show()