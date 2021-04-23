# Imports
import pyro
import torch
import numpy as np
from torch import nn
import pyro.distributions as dist
from scipy.integrate import nquad
from sklearn.cluster import KMeans
from pyro.ops.indexing import Vindex
from pyro.nn import PyroModule, PyroParam
from torch.distributions import transform_to
import pyro.distributions.constraints as constraints
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from torch.utils.data import RandomSampler, BatchSampler
from sklearn.covariance import EmpiricalCovariance

class GaussianMixtureModel(PyroModule):
	"""Gaussian Mixture Model following the CP formulation"""

	def __init__(self, K, M, device='cpu'):
		super().__init__()
		self.K = K
		self.M = M
		self.device = device
		self.kwargs = {'K': K, 'M': M, 'device': device}
		self.train_losses = list()
		self.val_losses = list()
		
		self.weights = PyroParam(
			(torch.ones(K, device=self.device) / K),
			constraint=constraints.simplex)
		
		self.random_init()

		if 'cuda' in self.device:
			self.cuda(self.device)
		
		self.eval()
	
	def random_init(self):
		self.locs = PyroParam(
			torch.randn(self.K, self.M, device=self.device),
			constraint=constraints.real)

		self.scales = PyroParam(
			torch.rand(self.K, self.M, device=self.device),
			constraint=constraints.positive)

	@config_enumerate
	def forward(self, data=None, n_samples=1000):
		N, M = data.shape if data is not None else (n_samples, self.M)

		# Empty tensor for data samples
		x_sample = torch.empty(N, M, device=self.device)

		with pyro.plate('data', size=N):
			# Sample cluster/component
			k = pyro.sample('k', dist.Categorical(self.weights))

			for m in range(self.M):
				# Observations of x_m
				obs = data[:, m] if data is not None else None

				# Sample x_m
				x_sample[:, m] = pyro.sample(
					f'x_{m}',
					dist.Normal(loc=self.locs[k, m],
								scale=self.scales[k, m]+1e-9),
					obs=obs)

			return x_sampleq
	
	def guide(self, data):
		pass

	def fit_model(self, data, data_val=None, lr=3e-4, mb_size=512,
		n_epochs=500, verbose=True, early_stopping=True):
		N_train = len(data)
		if data_val is not None:
			N_val = len(data_val)
		else:
			early_stopping = False
        
        # Variables for early-stopping
		best_loss = np.inf
		count = 0
		
		adam = pyro.optim.Adam({"lr": lr})
		svi = SVI(self, self.guide, adam, loss=TraceEnum_ELBO())
		
		for epoch in range(n_epochs):
			self.train()
			
			mbs = BatchSampler(
				RandomSampler(range(N_train)),
				batch_size=mb_size,
				drop_last=False)
			
			loss = 0
			for mb_idx in mbs:
				loss += svi.step(data[mb_idx])
			
			self.train_losses.append(loss/N_train)
			
			if data_val is not None:
				self.eval()
				val_loss = self.nllh(data_val) / N_val
				self.val_losses.append(val_loss)
			
			if epoch % 10 == 0 and verbose:
				print('[epoch {}]  loss: {:.4f}'.format(epoch, loss/N_train))
			
			if early_stopping:
				# Reset counter if val loss has improved by 0.1%
				if val_loss < best_loss*(1 - 1e-3):
					best_loss = val_loss
					count = 0
				else:
					count += 1
				
				# Break training loop if val loss has not improved in 10 epochs
				if count == 10:
					break

	def init_from_data(self, data, k_means=True):
		N, M = data.shape

		self.locs = PyroParam(
			data[torch.multinomial(torch.ones(N) / N, self.K),],
			constraint=constraints.real)
		
		self.scales = PyroParam(
			data.std(dim=0).repeat(self.K).reshape(self.K, self.M),
			constraint=constraints.positive)
		
		if k_means:
			k_means_model = KMeans(self.K)
			k_means_model.fit(data.cpu().numpy())
			locs = k_means_model.cluster_centers_

			self.locs = PyroParam(
				torch.tensor(locs, device=self.device),
				constraint=constraints.real)
	
	def log_density(self, data):
		with torch.no_grad():
			data = data.unsqueeze(1) # N x 1 x M
			log_weights = torch.log(self.weights) # K
			log_probs = dist.Normal(
				loc=self.locs,
				scale=self.scales).log_prob(data) # N x K x M
			log_probs = log_probs.sum(dim=-1) # N x K

			return torch.logsumexp(log_weights + log_probs, dim=-1) # N
	
	def nllh(self, data):
		with torch.no_grad():
			log_density = self.log_density(data)
			return -torch.sum(log_density).item()

	def density(self, data):
		with torch.no_grad():
			N, M = data.shape

			if M != self.M:
				raise ValueError('Incorrect number of data columns.')

			log_probs = dist.Normal(
				loc=self.locs,
				scale=self.scales).log_prob(data.unsqueeze(1))

			density = (
				torch.exp(log_probs.sum(dim=2)) * self.weights).sum(dim=1)

			return density

	def log_likelihood(self, data):
		with torch.no_grad():
			llh = torch.log(self.density(data)).sum()
			return llh

	def eval_density_grid(self, n_points=100, grid=[-5, 5, -5, 5]):
		x_range = np.linspace(grid[0], grid[1], n_points)
		y_range = np.linspace(grid[2], grid[3], n_points)
		X1, X2 = np.meshgrid(x_range, y_range)
		XX = np.column_stack((X1.ravel(), X2.ravel()))
		densities = self.density(
			torch.tensor(XX, device=self.device)).cpu().numpy()

		return (x_range, y_range), densities.reshape((n_points, n_points))


class CPModel(PyroModule):
	"""CP model where the distribution on each
	   variable can be specified"""

	def __init__(self, K, distributions, device='cpu'):
		super().__init__()
		self.K = K
		self.distributions = distributions
		self.device = device
		self.kwargs = {'K': K, 'distributions': distributions, 'device': device}
		self.M = len(distributions)
		self.train_losses = list()

		self.weights = PyroParam(
			torch.ones(self.K, device=self.device) / self.K,
			constraint=constraints.simplex)

		self.init_params()

		if 'cuda' in self.device:
			self.cuda(self.device)

		self.eval()

	def init_params(self):

		self.components = nn.ModuleList()

		for m, d in enumerate(self.distributions):
			# Create pyro module for attribute m (i.e. x_m)
			component = PyroModule(name=str(m))

			# Distribution of x_m
			component.dist = d

			# List for parameter names of distribution
			component.params = list()

			# Parameters and constraints of distribution
			for param, constr in d.arg_constraints.items():
				# Initialize parameter and set as an attribute of the component
				# Example:
				# component.loc = PyroParam(init_tensor, constraint=constraints.real)
				init_tensor = transform_to(constr)(
					torch.empty(self.K, device=self.device).uniform_(-1, 1))
				setattr(component, param, PyroParam(init_tensor, constraint=constr))

				# Add parameter name to list
				component.params.append(param)

			# Add component to module list
			self.components.append(component)

	@config_enumerate
	def forward(self, data=None, n_samples=1000):
		N, M = data.shape if data is not None else (n_samples, self.M)

		if M != self.M:
			raise ValueError('Incorrect number of data columns.')

		# Empty tensor for samples
		x_sample = torch.empty(N, M, device=self.device)

		with pyro.plate('data', N):
			# Draw cluster/component
			k = pyro.sample('k', dist.Categorical(self.weights))

			for m in range(M):
				# Observations of x_m
				obs = data[:, m] if data is not None else None

				# Parameters for distribution of x_m
				params = {param: getattr(self.components[m], param)[k]
						  for param in self.components[m].params}

				# Draw samples of x_m
				x_sample[:, m] = pyro.sample(
					f'x_{m}',
					self.components[m].dist(**params),
					obs=obs)

			return x_sample
	
	def guide(self, data):
		pass

	def fit_model(self, data, lr=3e-4, n_steps=10000, verbose=True):
		self.train()

		adam = pyro.optim.Adam({"lr": lr})
		svi = SVI(self, self.guide, adam, loss=TraceEnum_ELBO())

		for step in range(n_steps):
			loss = svi.step(data)
			self.train_losses.append(loss)

			if step % 1000 == 0 and verbose:
				print('[iter {}]  loss: {:.4f}'.format(step, loss))

		self.eval()

	def hot_start(self, data, n_starts=100):
		seeds = torch.multinomial(
			torch.ones(10000) / 10000, num_samples=n_starts)
		inits = list()

		for seed in seeds:
			pyro.set_rng_seed(seed)
			pyro.clear_param_store()

			# Set new initial parameters
			self.init_params()

			# Get initial loss
			self.fit_model(data, lr=0, n_steps=1, verbose=False)
			loss = self.train_losses[-1]

			# Save loss and seed
			inits.append((loss, seed))

			# Reset train losses
			self.train_losses = list()

		# Best initialization
		_, best_seed = min(inits)

		# Initialize with best seed
		pyro.set_rng_seed(best_seed)
		pyro.clear_param_store()
		self.init_params()

	def density(self, data):
		with torch.no_grad():
			N, M = data.shape

			if M != self.M:
				raise ValueError('Incorrect number of data columns.')

			log_probs = torch.empty(N, self.K, M, device=self.device)

			for m in range(M):
				params = {param: getattr(self.components[m], param)
						  for param in self.components[m].params}

				log_probs[:, :, m] = self.components[m].dist(
					**params).log_prob(data[:, m].unsqueeze(1))

			density = (
					torch.exp(log_probs.sum(dim=2)) * self.weights).sum(dim=1)

			return density

	def log_likelihood(self, data):
		with torch.no_grad():
			llh = torch.log(self.density(data)).sum()
			return llh

	def eval_density_grid(self, n_points=100):
		x_range = np.linspace(-5, 5, n_points)
		X1, X2 = np.meshgrid(x_range, x_range)
		XX = np.column_stack((X1.ravel(), X2.ravel()))
		densities = self.density(
			torch.tensor(XX, device=self.device)).cpu().numpy()

		return x_range, densities.reshape((n_points, n_points))

	def unit_test(self, int_limits, opts=dict()):
		"""Integrates probablity density"""
		return nquad(
			lambda *args: self.density(
				torch.tensor([args], device=self.device)).item(),
			int_limits, opts=opts)


class TensorTrain(PyroModule):
	"""Tensor Train model"""

	def __init__(self, Ks, device='cpu'):
		super().__init__()
		self.Ks = Ks
		self.device = device
		self.kwargs = {'Ks': Ks, 'device': device}
		self.M = len(self.Ks) - 1
		self.train_losses = list()
		self.val_losses = list()

		# Weights for latent variable k_0
		self.k0_weights = PyroParam(
			torch.ones(self.Ks[0], device=self.device) / self.Ks[0],
			constraint=dist.constraints.simplex)

		# Parameters indexed by latent variable number
		# I.e. by 1, 2,..., M
		self.params = nn.ModuleDict()

		# Intialize weights, locs and scales
		self.init_params()

		if 'cuda' in self.device:
			self.cuda(self.device)
		
		if len(self.Ks) > 60:
			self.forward = self.forward_alt
		
	def init_params(self, loc_min=None, loc_max=None, scale_max=None):

		if loc_min is None:
			loc_min = [-1 for m in range(self.M)]
		if loc_max is None:
			loc_max = [1 for m in range(self.M)]
		if scale_max is None:
			scale_max = [1 for m in range(self.M)]

		for m in range(1, self.M + 1):
			# PyroModule for storing weights, locs
			# and scales for x_m
			module = PyroModule(name=f'x_{m}')
			param_shape = (self.Ks[m-1], self.Ks[m])
			
			# Weight matrix W^m ("Transition probabilities")
			# I.e. probability of k_m = a given k_{m-1} = b
			module.weights = PyroParam(
				torch.ones(param_shape, device=self.device) / self.Ks[m],
				constraint=constraints.simplex)
			
			# Locs for x_m
			module.locs = PyroParam(
				dist.Uniform(loc_min[m-1]-1e-8, loc_max[m-1]+1e-8).sample(
					param_shape).to(self.device),
				constraint=constraints.real)
			
			# Scales for x_m
			module.scales = PyroParam(
				dist.Uniform(1e-8, scale_max[m-1]+1e-7).sample(
					param_shape).to(self.device),
				constraint=constraints.positive)
			
			self.params[str(m)] = module
	
	@config_enumerate
	def forward(self, data=None, n_samples=1000):
		N, M = data.shape if data is not None else (n_samples, self.M)

		# Empty tensor for data samples
		x_sample = torch.empty(N, M, device=self.device)

		with pyro.plate('data', size=N):
			# Sample k_0
			k_m_prev = pyro.sample(
				'k_0',
				dist.Categorical(self.k0_weights))

			for m, params in enumerate(self.params.values()):
				# Sample k_m
				k_m = pyro.sample(
					f'k_{m + 1}',
					dist.Categorical(Vindex(params.weights)[k_m_prev]))

				# Observations of x_m
				obs = data[:, m] if data is not None else None

				# Sample x_m
				x_sample[:, m] = pyro.sample(
					f'x_{m + 1}',
					dist.Normal(loc=Vindex(params.locs)[k_m_prev, k_m],
								scale=Vindex(params.scales)[k_m_prev, k_m]+1e-9),
								obs=obs)

				k_m_prev = k_m

			return x_sample
	
	@config_enumerate
	def forward_alt(self, data=None, n_samples=1000):
		N, M = data.shape if data is not None else (n_samples, self.M)

		# Empty tensor for data samples
		x_sample = torch.empty(N, M, device=self.device)

		with pyro.plate('data', size=N):
			# Sample k_0
			k_m_prev = pyro.sample(
				'k_0',
				dist.Categorical(self.k0_weights))

			for m in pyro.markov(range(self.M)):
				params = self.params[str(m +1)]
				# Sample k_m
				k_m = pyro.sample(
					f'k_{m + 1}',
					dist.Categorical(Vindex(params.weights)[k_m_prev]))

				# Observations of x_m
				obs = data[:, m] if data is not None else None

				# Sample x_m
				x_sample[:, m] = pyro.sample(
					f'x_{m + 1}',
					dist.Normal(loc=Vindex(params.locs)[k_m_prev, k_m],
								scale=Vindex(params.scales)[k_m_prev, k_m]+1e-9),
								obs=obs)

				k_m_prev = k_m

			return x_sample

	def guide(self, data):
		pass

	def fit_model(self, data, data_val=None, lr=3e-4, mb_size=512,
		n_epochs=500, verbose=True, early_stopping=True):
		N_train = len(data)
		if data_val is not None:
			N_val = len(data_val)
		else:
			early_stopping = False
		
		# Variables for early-stopping
		best_loss = np.inf
		count = 0

		adam = pyro.optim.Adam({"lr": lr})
		svi = SVI(self, self.guide, adam, loss=TraceEnum_ELBO())

		for epoch in range(n_epochs):
			self.train()

			mbs = BatchSampler(
				RandomSampler(range(N_train)),
				batch_size=mb_size,
				drop_last=False)

			loss = 0
			for mb_idx in mbs:
				loss += svi.step(data[mb_idx])
			
			self.train_losses.append(loss/N_train)
			
			if data_val is not None:
				self.eval()
				val_loss = self.nllh(data_val) / N_val
				self.val_losses.append(val_loss)

			if epoch % 10 == 0 and verbose:
				print('[epoch {}]  loss: {:.4f}'.format(epoch, loss/N_train))

			if early_stopping:
				# Reset counter if val loss has improved by 0.1%
				if val_loss < best_loss*(1 - 1e-3):
					best_loss = val_loss
					count = 0
				else:
					count += 1
				
				# Break training loop if val loss has not improved in 10 epochs
				if count == 10:
					break
				
	def hot_start(self, data, subsample_size=None, n_starts=100):
		seeds = torch.multinomial(
			torch.ones(10000) / 10000, num_samples=n_starts)
		inits = list()

		data_min = data.min(dim=0).values
		data_max = data.max(dim=0).values
		data_std = data.std(dim=0)

		if subsample_size is not None:
			subsample_idx = torch.randperm(len(data))[:subsample_size]
			data = data[subsample_idx]
	
		for seed in seeds:
			pyro.set_rng_seed(seed)
			pyro.clear_param_store()

			# Set new initial parameters
			self.init_params(loc_min=data_min,
							 loc_max=data_max,
							 scale_max=data_std)

			# Get initial loss
			self.fit_model(
				data, lr=0, mb_size=len(data), n_epochs=1, verbose=False)
			loss = self.train_losses[-1]

			# Save loss and seed
			inits.append((loss, seed))

			# Reset train losses
			self.train_losses = list()

		# Best initialization
		_, best_seed = min(inits)

		# Initialize with best seed
		pyro.set_rng_seed(best_seed)
		pyro.clear_param_store()
		self.init_params(loc_min=data_min,
						 loc_max=data_max,
						 scale_max=data_std)

	def log_density(self, data):
		with torch.no_grad():
			N, M = data.shape
			data = data.reshape(N, M, 1, 1) # N x M x 1 x 1
			
			# Initialize log density
			log_density = torch.zeros(1,1,1, device=self.device) # 1 x 1 x 1
			
			for m, params in reversed(self.params.items()):
				m = int(m) - 1
				log_prob = dist.Normal(
					params.locs, params.scales).log_prob(data[:, m]) # N x K_{m-1} x K_m
				log_weights = torch.log(params.weights).unsqueeze(0) # 1 x K_{m-1} x K_m
				log_density = torch.logsumexp(
					log_weights + log_prob + log_density, dim=-1).unsqueeze(1) # N x 1 x K_{m-1}
				
			log_density = log_density.squeeze(1) # N x K_0
			log_k0_weights = torch.log(self.k0_weights) # K_0
			
			return torch.logsumexp(log_k0_weights + log_density, dim=-1) # N
	
	def nllh(self, data):
		with torch.no_grad():
			log_density = self.log_density(data)
			return -torch.sum(log_density).item()

	def density(self, data):
		with torch.no_grad():
			N, M = data.shape

			# Intialize sum to neutral multiplier
			sum_ = torch.ones(1, 1, device=self.device)

			# Iterate in reverse order
			# (to sum last latent variables first)
			for m, params in reversed(self.params.items()):
				# Modules are indexed by 1,2,...,M
				# but data by 0,1,...,M-1
				m = int(m) - 1

				probs = torch.exp(dist.Normal(
					loc=params.locs,
					scale=params.scales).log_prob(data[:, m].reshape(-1, 1, 1)))

				sum_ = (params.weights * probs * sum_.unsqueeze(1)).sum(-1)

			density = (self.k0_weights * sum_).sum(-1)

			return density

	def log_likelihood(self, data):
		with torch.no_grad():
			llh = torch.log(self.density(data)).sum()
			
			return llh

	def eval_density_grid(self, n_points=100, grid=[-5, 5, -5, 5]):
		x_range = np.linspace(grid[0], grid[1], n_points)
		y_range = np.linspace(grid[2], grid[3], n_points)
		X1, X2 = np.meshgrid(x_range, y_range)
		XX = np.column_stack((X1.ravel(), X2.ravel()))
		densities = self.density(
			torch.tensor(XX, device=self.device)).cpu().numpy()

		return (x_range, y_range), densities.reshape((n_points, n_points))

	def unit_test(self, int_limits, opts=dict()):
		"""Integrates probablity density"""
		return nquad(
			lambda *args: self.density(
				torch.tensor([args], device=self.device)).item(),
			int_limits, opts=opts)

	def unit_test_alt(self, n_points=100, grid=[-5, 5, -5, 5]):
		(x_range, y_range), density = self.eval_density_grid(n_points=n_points,
															 grid=grid)
		density = density.sum()
		height = (y_range[-1] - y_range[0])/len(y_range)
		width  = (x_range[-1] - x_range[0])/len(x_range)
		scaled_density = density*width*height

		return scaled_density

	def eval_density_multi_grid(self, limits, n_points):
		# Reshape limits
		limits = torch.Tensor(limits).reshape(self.M, 2)
		# Allocate interval ranges
		grid_tensor = torch.zeros(self.M, n_points)
		# Fill in interval ranges
		for m, vals in enumerate(limits):
			grid_tensor[m, :] = torch.linspace(vals[0], vals[1], n_points)

		# Create meshgrid of points
		mesh = torch.meshgrid([grid for grid in grid_tensor])
		XX = torch.column_stack([x.ravel() for x in mesh])

		# Evaluate densities
		densities = self.density(
			torch.tensor(XX, device=self.device)).cpu().numpy()

		# Return ranges for M dimensions and all densities

		if self.M == 2:
			return (grid_tensor[0].numpy(), grid_tensor[1].numpy()), densities.reshape((n_points, n_points))
		else:
			return grid_tensor, densities

	def unit_test_multidimensional(self, limits, n_points=1000):
		grid_tensor, densities = self.eval_density_multi_grid(limits, n_points)
		total_density = densities.sum()
		if self.M == 2:
			scaling = (grid_tensor[0][-1]-grid_tensor[0][0])*(grid_tensor[1][-1]-grid_tensor[1][0])/(n_points**self.M)
		else:
			scaling = np.prod([(vals[-1]-vals[0]).item() for vals in grid_tensor])/(n_points**self.M)
		scaled_density = total_density * scaling
		return scaled_density

class GaussianMixtureModelFull(PyroModule):
	"""Gaussian Mixture Model with full covariance matrix"""

	def __init__(self, K, device='cpu'):
		super().__init__()
		self.device = device
		self.M = len(K)-1 # To comply with other models initialisations
		self.K = K[0]	  # To comply with other models initialisations
		self.locs = None
		self.scales = None      # Variances
		self.cov    = None      # Covariances
		self.scale_tril = None  # Covariance matrix
		self.initialized = False
		self.weights = PyroParam(
			torch.ones(self.K, device=self.device) / self.K,
			constraint=dist.constraints.simplex)
		self.kwargs = {'K': self.K, 'M': self.M, 'device': device}
		self.train_losses = list()
		self.val_losses = list()
		
		if 'cuda' in self.device:
			self.cuda(self.device)

		self.eval()

	@config_enumerate
	def forward(self, data=None):
		N, M = data.shape if data is not None else (1000, 2)

		if self.initialized is False and data is not None:
			self.init_from_data(data)
		elif self.initialized is False:
			self.random_init(M)

		if M != self.M:
			raise ValueError('Incorrect number of data columns.')

		# Combine covariances and variances
		for k in range(self.K):
			self.scale_tril[k, :, :] = torch.mm(self.scales[k,:].diag_embed(), self.cov[k, :, :])


		# Empty tensor for data samples
		x_sample = torch.empty(N, M)

		with pyro.plate('data', N):
			# Sample cluster/component
			k = pyro.sample('k', dist.Categorical(self.weights))

			x_sample = pyro.sample('x_sample', dist.MultivariateNormal(loc=self.locs[k], scale_tril=self.scale_tril[k]),
								   obs=data)



			return x_sample

	def init_from_data(self, data, k_means=False):
		N, M = data.shape
		self.M = M

		self.locs = PyroParam(
			data[torch.multinomial(torch.ones(N, device=self.device) / N, self.K),],
			constraint=constraints.real, )

		self.scales = PyroParam(
			data.std(dim=0).repeat(self.K).reshape(self.K, self.M).to(self.device),
			constraint=constraints.positive)

		if k_means:
			k_means_model = KMeans(self.K)
			k_means_model.fit(data.cpu().numpy())
			locs = k_means_model.cluster_centers_

			self.locs = PyroParam(
				torch.tensor(locs, device=self.device),
				constraint=constraints.real)

		# Initialisation of lower cholesky factorisation of covariance matrix
		cov = torch.zeros(self.K, self.M, self.M)
		for k in range(self.K):
			cov[k, :, :] =dist.LKJCholesky(self.M).sample()
		self.cov = PyroParam(cov.to(device=self.device), constraint=constraints.lower_cholesky, event_dim=2)

		# Covariance matrix placeholder
		self.scale_tril = torch.zeros(self.K, self.M, self.M, device=self.device)

		# Combination of to lower lower triangular covariance matrix (scale_tril)
		for k in range(self.K):
			self.scale_tril[k, :, :] = torch.mm(self.scales[k,:].diag_embed(), self.cov[k, :, :])

		self.initialized = True

	def random_init(self, M):
		self.M = M

		self.locs = PyroParam(
			torch.rand(self.K, self.M, device=self.device),
			constraint=constraints.real)

		self.scales = PyroParam(
			torch.rand(self.K, self.M, device=self.device),
			constraint=constraints.positive)

		# Initialisation of lower cholesky factor of covariance matrix
		cov = torch.zeros(self.K, self.M, self.M)
		for k in range(self.K):
			cov[k, :, :] =dist.LKJCholesky(self.M).sample()
		self.cov = PyroParam(cov.to(device=self.device), constraint=constraints.lower_cholesky, event_dim=2)

		# Covariance matrix placeholder
		self.scale_tril = torch.zeros(self.K, self.M, self.M, device=self.device)

		# Combination of to lower lower triangular covariance matrix (scale_tril)
		for k in range(self.K):
			self.scale_tril[k, :, :] = torch.mm(self.scales[k,:].diag_embed(), self.cov[k, :, :])

		self.initialized = True

	def guide(self, data):
		pass

	def fit_model(self, data, data_val=None, lr=3e-4, mb_size=512,
		n_epochs=500, verbose=True, early_stopping=True):
		N_train = len(data)
		if data_val is not None:
			N_val = len(data_val)
		else:
			early_stopping = False
		
		# Variables for early-stopping
		best_loss = np.inf
		count = 0

		adam = pyro.optim.Adam({"lr": lr})
		svi = SVI(self, self.guide, adam, loss=TraceEnum_ELBO())

		for epoch in range(n_epochs):
			self.train()

			mbs = BatchSampler(
				RandomSampler(range(N_train)),
				batch_size=mb_size,
				drop_last=False)

			loss = 0
			for mb_idx in mbs:
				loss += svi.step(data[mb_idx])
			
			self.train_losses.append(loss/N_train)
			
			if data_val is not None:
				self.eval()
				val_loss = self.nllh(data_val) / N_val
				self.val_losses.append(val_loss)

			if epoch % 10 == 0 and verbose:
				print('[epoch {}]  loss: {:.4f}'.format(epoch, loss/N_train))

			if early_stopping:
				# Reset counter if val loss has improved by 0.1%
				if val_loss < best_loss*(1 - 1e-3):
					best_loss = val_loss
					count = 0
				else:
					count += 1
				
				# Break training loop if val loss has not improved in 10 epochs
				if count == 10:
					break

	def density(self, data):
		with torch.no_grad():
			N, M = data.shape

			if M != self.M:
				raise ValueError('Incorrect number of data columns.')

			density = torch.zeros(data.shape[0], device=self.device)
			for k in range(self.K):
				density += self.weights[k]* torch.exp(dist.MultivariateNormal(loc=self.locs[k].double(),
													   scale_tril=self.scale_tril[k].double()).log_prob(data))
			return density

	def log_likelihood(self, data):
		with torch.no_grad():
			llh = torch.log(self.density(data)).sum()

			return llh

	def nllh(self, data):
		return -self.log_likelihood(data)

	def eval_density_grid(self, n_points=100, grid=[-5, 5, -5, 5]):
		if self.M != 2:
			raise ValueError('Can only evaluate density grid for 2-dimensional data.')
		x_range = np.linspace(grid[0], grid[1], n_points)
		y_range = np.linspace(grid[2], grid[3], n_points)
		X1, X2 = np.meshgrid(x_range, y_range)
		XX = np.column_stack((X1.ravel(), X2.ravel()))
		densities = self.density(
			torch.tensor(XX, device=self.device)).cpu().numpy()

		return (x_range, y_range), densities.reshape((n_points, n_points))

	def unit_test(self, int_limits, opts=dict()):
		"""Integrates probablity density"""
		return nquad(
			lambda *args: self.density(
				torch.tensor([args], device=self.device)).item(),
			int_limits, opts=opts)

	def unit_test_alt(self, n_points=100, grid=[-5, 5, -5, 5]):
		(x_range, y_range), density = self.eval_density_grid(n_points=n_points,
															 grid=grid)
		density = density.sum()
		height = (y_range[-1] - y_range[0]) / len(y_range)
		width = (x_range[-1] - x_range[0]) / len(x_range)
		scaled_density = density * width * height

		return scaled_density

	def eval_density_multi_grid(self, limits, n_points):
		# Reshape limits
		limits = torch.Tensor(limits).reshape(self.M, 2)
		# Allocate interval ranges
		grid_tensor = torch.zeros(self.M, n_points, device=self.device)
		# Fill in interval ranges
		for m, vals in enumerate(limits):
			grid_tensor[m, :] = torch.linspace(vals[0], vals[1], n_points)

		# Create meshgrid of points
		mesh = torch.meshgrid([grid for grid in grid_tensor])
		XX = torch.column_stack([x.ravel() for x in mesh])
		# Evaluate densities
		densities = self.density(
			torch.tensor(XX, device=self.device)).cpu().detach().numpy()

		# Return ranges for M dimensions and all densities

		if self.M == 2:
			return (grid_tensor[0].cpu().numpy(), grid_tensor[1].cpu().numpy()), densities.reshape((n_points, n_points))
		else:
			return grid_tensor, densities

	def unit_test_multidimensional(self, limits, n_points=1000):
		grid_tensor, densities = self.eval_density_multi_grid(limits, n_points)
		total_density = densities.sum()
		if self.M == 2:
			scaling = (grid_tensor[0][-1] - grid_tensor[0][0]) * (grid_tensor[1][-1] - grid_tensor[1][0]) / (
						n_points ** self.M)
		else:
			scaling = np.prod([(vals[-1] - vals[0]).item() for vals in grid_tensor]) / (n_points ** self.M)
		scaled_density = total_density * scaling
		return scaled_density

class GaussianMixtureModelFullAlt(PyroModule):
    """Gaussian Mixture Model with full covariance matrix"""

    def __init__(self, K, M, device='cpu'):
        super().__init__()
        self.device = device
        self.K = K
        self.M = M
        self.kwargs = {'K': self.K, 'M': self.M, 'device': self.device}
        self.train_losses = list()
        self.val_losses = list()
        
        if 'cuda' in self.device:
            self.cuda(self.device)
        
        self.random_init()
        self.eval()
        
    def random_init(self):
        
        self.weights = PyroParam(
            torch.ones(self.K, device=self.device) / self.K,
            constraint=dist.constraints.simplex)
        
        self.locs = PyroParam(
            torch.rand(self.K, self.M, device=self.device),
            constraint=constraints.real)
        
        self.scale_tril = PyroParam(
            dist.LKJCholesky(self.M).sample((self.K,)).to(self.device),
            constraint=constraints.lower_cholesky)
    
    @config_enumerate
    def forward(self, data=None):
        N, M = data.shape if data is not None else (1000, self.M)

        with pyro.plate('data', N):
            # Sample cluster/component
            k = pyro.sample('k', dist.Categorical(self.weights))

            x_sample = pyro.sample('x_sample',
                                   dist.MultivariateNormal(
                                       loc=Vindex(self.locs)[k],
                                       scale_tril=Vindex(self.scale_tril)[k]),
                                   obs=data)

            return x_sample

    def init_from_data(self, data):
        data_np = data.cpu().numpy()
        
        k_means_model = KMeans(self.K)
        k_means_model.fit(data_np)
        locs = k_means_model.cluster_centers_
        labels = k_means_model.predict(data_np)
        
        self.locs = PyroParam(
            torch.tensor(locs, device=self.device),
            constraint=constraints.real)
        
        weights = torch.zeros(self.K)
        scale_tril = torch.zeros(self.K, self.M, self.M)

        for k in range(self.K):
            weights[k] = np.sum(labels==k) / len(data_np)
            cov = EmpiricalCovariance(store_precision=False).fit(data_np[labels==k])
            try:
                L = np.linalg.cholesky(cov.covariance_)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(np.eye(self.M))
            scale_tril[k,:,:] = torch.tensor(L)
        
        self.weights = PyroParam(
            weights.to(self.device),
            constraint=dist.constraints.simplex)
        
        self.scale_tril = PyroParam(
            scale_tril.to(self.device),
            constraint=constraints.lower_cholesky)

    def guide(self, data):
        pass

    def fit_model(self, data, data_val=None, lr=3e-4, mb_size=512,
        n_epochs=500, verbose=True, early_stopping=True):
        N_train = len(data)
        if data_val is not None:
            N_val = len(data_val)
        else:
            early_stopping = False
        
        # Variables for early-stopping
        best_loss = np.inf
        count = 0

        adam = pyro.optim.Adam({"lr": lr})
        svi = SVI(self, self.guide, adam, loss=TraceEnum_ELBO())

        for epoch in range(n_epochs):
            self.train()

            mbs = BatchSampler(
                RandomSampler(range(N_train)),
                batch_size=mb_size,
                drop_last=False)

            loss = 0
            for mb_idx in mbs:
                loss += svi.step(data[mb_idx])
            
            self.train_losses.append(loss/N_train)
            
            if data_val is not None:
                self.eval()
                val_loss = self.nllh(data_val) / N_val
                self.val_losses.append(val_loss)

            if epoch % 10 == 0 and verbose:
                print('[epoch {}]  loss: {:.4f}'.format(epoch, loss/N_train))

            if early_stopping:
                # Reset counter if val loss has improved by 0.1%
                if val_loss < best_loss*(1 - 1e-3):
                    best_loss = val_loss
                    count = 0
                else:
                    count += 1
                
                # Break training loop if val loss has not improved in 10 epochs
                if count == 10:
                    break
    
    def log_density(self, data):
        with torch.no_grad():
            N, M = data.shape
            data = data.reshape(N, 1, M)
            
            log_weights = torch.log(self.weights)
            log_prob = dist.MultivariateNormal(
                loc=self.locs,
                scale_tril=self.scale_tril).log_prob(data)
            
            return torch.logsumexp(log_weights + log_prob, dim=-1)
    
    def nllh(self, data):
        with torch.no_grad():
            log_density = self.log_density(data)
            return -torch.sum(log_density).item()

    def density(self, data):
        with torch.no_grad():
            N, M = data.shape

            if M != self.M:
                raise ValueError('Incorrect number of data columns.')

            density = torch.zeros(data.shape[0], device=self.device)
            for k in range(self.K):
                density += self.weights[k]* torch.exp(dist.MultivariateNormal(loc=self.locs[k].double(),
                                                       scale_tril=self.scale_tril[k].double()).log_prob(data))
            return density

    def log_likelihood(self, data):
        with torch.no_grad():
            llh = torch.log(self.density(data)).sum()

            return llh

    def eval_density_grid(self, n_points=100, grid=[-5, 5, -5, 5]):
        if self.M != 2:
            raise ValueError('Can only evaluate density grid for 2-dimensional data.')
        x_range = np.linspace(grid[0], grid[1], n_points)
        y_range = np.linspace(grid[2], grid[3], n_points)
        X1, X2 = np.meshgrid(x_range, y_range)
        XX = np.column_stack((X1.ravel(), X2.ravel()))
        densities = self.density(
            torch.tensor(XX, device=self.device)).cpu().numpy()

        return (x_range, y_range), densities.reshape((n_points, n_points))

    def unit_test_alt(self, n_points=100, grid=[-5, 5, -5, 5]):
        (x_range, y_range), density = self.eval_density_grid(n_points=n_points,
                                                             grid=grid)
        density = density.sum()
        height = (y_range[-1] - y_range[0]) / len(y_range)
        width = (x_range[-1] - x_range[0]) / len(x_range)
        scaled_density = density * width * height

        return scaled_density


class TensorRing(PyroModule):
	"""Tensor Train model"""

	def __init__(self, Ks, device='cpu'):
		super().__init__()
		self.Ks = Ks
		self.device = device
		self.kwargs = {'Ks': Ks, 'device': device}
		self.M = len(self.Ks) - 1
		self.train_losses = list()
		self.val_losses = list()

		# Weights for latent variable k_0
		self.k0_weights = PyroParam(
			torch.ones(self.Ks[0], device=self.device) / self.Ks[0],
			constraint=dist.constraints.simplex)

		# Parameters indexed by latent variable number
		# I.e. by 1, 2,..., M
		self.params = nn.ModuleDict()

		# Intialize weights, locs and scales
		self.init_params()

		if 'cuda' in self.device:
			self.cuda(self.device)
		
		if len(self.Ks) > 60:
			self.forward = self.forward_alt
		
	def init_params(self, loc_min=None, loc_max=None, scale_max=None):

		if loc_min is None:
			loc_min = [-1 for m in range(self.M)]
		if loc_max is None:
			loc_max = [1 for m in range(self.M)]
		if scale_max is None:
			scale_max = [1 for m in range(self.M)]

		# Initialise parameters for the k_{1} to k_{M-1}. k_{M} is treated separately.
		for m in range(1, self.M):
			# PyroModule for storing weights, locs
			# and scales for x_m
			module = PyroModule(name=f'x_{m}')
			param_shape = (self.Ks[m-1], self.Ks[m])
			
			# Weight matrix W^m ("Transition probabilities")
			# I.e. probability of k_m = a given k_{m-1} = b
			module.weights = PyroParam(
				torch.ones(param_shape, device=self.device) / self.Ks[m],
				constraint=constraints.simplex)
			
			# Locs for x_m
			module.locs = PyroParam(
				dist.Uniform(loc_min[m-1]-1e-8, loc_max[m-1]+1e-8).sample(
					param_shape).to(self.device),
				constraint=constraints.real)
			
			# Scales for x_m
			module.scales = PyroParam(
				dist.Uniform(1e-8, scale_max[m-1]+1e-7).sample(
					param_shape).to(self.device),
				constraint=constraints.positive)
			
			self.params[str(m)] = module
		
		# Initialise parameters for k_{M}
		module = PyroModule(name=f'x_{self.M}')
		param_shape = (self.Ks[self.M-1], self.Ks[self.M])

		# No weights for k_{M}
		
		# Locs for x_m
		module.locs = PyroParam(
			dist.Uniform(loc_min[self.M-1]-1e-8, loc_max[self.M-1]+1e-8).sample(
				param_shape).to(self.device),
			constraint=constraints.real)
		
		# Scales for x_m
		module.scales = PyroParam(
			dist.Uniform(1e-8, scale_max[self.M-1]+1e-7).sample(
				param_shape).to(self.device),
			constraint=constraints.positive)
		
		self.params[str(self.M)] = module
	
	@config_enumerate
	def forward(self, data=None, n_samples=1000):
		N, M = data.shape if data is not None else (n_samples, self.M)

		# Empty tensor for data samples
		x_sample = torch.empty(N, M, device=self.device)

		with pyro.plate('data', size=N):
			# Sample k_0
			k_m_prev = pyro.sample(
				'k_0',
				dist.Categorical(self.k0_weights))
			k_init = k_m_prev

			for m, params in enumerate(self.params.values()):

				# Observations of x_m
				obs = data[:, m] if data is not None else None

				if m == self.M-1:
					 k_m = k_init
							
				else:
					# Sample k_m
					k_m = pyro.sample(
						f'k_{m + 1}',
						dist.Categorical(Vindex(params.weights)[k_m_prev]))

				# Sample x_m
				x_sample[:, m] = pyro.sample(
					f'x_{m + 1}',
					dist.Normal(loc=Vindex(params.locs)[k_m_prev, k_m],
								scale=Vindex(params.scales)[k_m_prev, k_m]),
								obs=obs)

				k_m_prev = k_m

			return x_sample
	
	@config_enumerate
	def forward_alt(self, data=None, n_samples=1000):
		N, M = data.shape if data is not None else (n_samples, self.M)

		# Empty tensor for data samples
		x_sample = torch.empty(N, M, device=self.device)

		with pyro.plate('data', size=N):
			# Sample k_0
			k_m_prev = pyro.sample(
				'k_0',
				dist.Categorical(self.k0_weights))

			for m in pyro.markov(range(self.M)):
				params = self.params[str(m +1)]
								# Sample k_m
				
				# Observations of x_m
				obs = data[:, m] if data is not None else None

				if m == self.M-1:
					# Sample x_m
					x_sample[:, m] = pyro.sample(
						f'x_{m + 1}',
						dist.Normal(loc=Vindex(params.locs)[k_m_prev, k_init],
									scale=Vindex(params.scales)[k_m_prev, k_init]),
									obs=obs)

				else:
					# Sample k_m
					k_m = pyro.sample(
					f'k_{m + 1}',
					dist.Categorical(Vindex(params.weights)[k_m_prev]))

					# Sample x_m
					x_sample[:, m] = pyro.sample(
						f'x_{m + 1}',
						dist.Normal(loc=Vindex(params.locs)[k_m_prev, k_m],
									scale=Vindex(params.scales)[k_m_prev, k_m]),
									obs=obs)

					k_m_prev = k_m

			return x_sample

	def guide(self, data):
		pass

	def fit_model(self, data, data_val=None, lr=3e-4, mb_size=512,
		n_epochs=500, verbose=True):
		N_train = len(data)
		if data_val is not None:
			N_val = len(data_val)

		adam = pyro.optim.Adam({"lr": lr})
		svi = SVI(self, self.guide, adam, loss=TraceEnum_ELBO())

		for epoch in range(n_epochs):
			self.train()

			mbs = BatchSampler(
				RandomSampler(range(N_train)),
				batch_size=mb_size,
				drop_last=False)

			loss = 0
			for mb_idx in mbs:
				loss += svi.step(data[mb_idx])
			
			self.train_losses.append(loss/N_train)
			
			if data_val is not None:
				self.eval()
				self.val_losses.append(svi.evaluate_loss(data_val)/N_val)

			if (epoch % 1000 == 0 or epoch==n_epochs) and verbose:
				print('[epoch {}]  loss: {:.4f}'.format(epoch, loss/N_train))

	def hot_start(self, data, subsample_size=None, n_starts=100):
		seeds = torch.multinomial(
			torch.ones(10000) / 10000, num_samples=n_starts)
		inits = list()

		data_min = data.min(dim=0).values
		data_max = data.max(dim=0).values
		data_std = data.std(dim=0)

		if subsample_size is not None:
			subsample_idx = torch.randperm(len(data))[:subsample_size]
			data = data[subsample_idx]
	
		for seed in seeds:
			pyro.set_rng_seed(seed)
			pyro.clear_param_store()

			# Set new initial parameters
			self.init_params(loc_min=data_min,
							 loc_max=data_max,
							 scale_max=data_std)

			# Get initial loss
			self.fit_model(
				data, lr=0, mb_size=len(data), n_epochs=1, verbose=False)
			loss = self.train_losses[-1]

			# Save loss and seed
			inits.append((loss, seed))

			# Reset train losses
			self.train_losses = list()

		# Best initialization
		_, best_seed = min(inits)

		# Initialize with best seed
		pyro.set_rng_seed(best_seed)
		pyro.clear_param_store()
		self.init_params(loc_min=data_min,
						 loc_max=data_max,
						 scale_max=data_std)

	def density(self, data):
		with torch.no_grad():
			N, M = data.shape

			# Intialize sum to neutral multiplier
			sum_ = torch.ones(1, 1, device=self.device)

			# Iterate in reverse order
			# (to sum last latent variables first)
			for m, params in reversed(self.params.items()):
				# Modules are indexed by 1,2,...,M
				# but data by 0,1,...,M-1
				m = int(m) - 1

				if m == self.M-1:
					end_params = params
					continue
				elif m == self.M-2:
					logprobs = (dist.Normal(
							loc=params.locs,
							scale=params.scales).log_prob(data[:, m].reshape(-1, 1, 1)))
					logprobs_prev = (dist.Normal(
							loc=end_params.locs,
							scale=end_params.scales).log_prob(data[:, m+1].reshape(-1, 1, 1)))
					sum_ = (params.weights * torch.exp( logprobs + logprobs_prev) * sum_.unsqueeze(1)).sum(-1)
				else:
					probs = torch.exp(dist.Normal(
						loc=params.locs,
						scale=params.scales).log_prob(data[:, m].reshape(-1, 1, 1)))

					sum_ = (params.weights * probs * sum_.unsqueeze(1)).sum(-1)

			density = (self.k0_weights * sum_).sum(-1)

			return density

	def log_likelihood(self, data):
		with torch.no_grad():
			llh = (torch.log(self.density(data))).sum()
			return llh

	def eval_density_grid(self, n_points=100, grid=[-5, 5, -5, 5]):
		x_range = np.linspace(grid[0], grid[1], n_points)
		y_range = np.linspace(grid[2], grid[3], n_points)
		X1, X2 = np.meshgrid(x_range, y_range)
		XX = np.column_stack((X1.ravel(), X2.ravel()))
		densities = self.density(
			torch.tensor(XX, device=self.device)).cpu().numpy()

		return (x_range, y_range), densities.reshape((n_points, n_points))

	def unit_test(self, int_limits, opts=dict()):
		"""Integrates probablity density"""
		return nquad(
			lambda *args: self.density(
				torch.tensor([args], device=self.device)).item(),
			int_limits, opts=opts)

	def unit_test_alt(self, n_points=100, grid=[-5, 5, -5, 5]):
		(x_range, y_range), density = self.eval_density_grid(n_points=n_points,
															 grid=grid)
		density = density.sum()
		height = (y_range[-1] - y_range[0])/len(y_range)
		width  = (x_range[-1] - x_range[0])/len(x_range)
		scaled_density = density*width*height40

		return scaled_density

	def eval_density_multi_grid(self, limits, n_points):
		# Reshape limits
		limits = torch.Tensor(limits).reshape(self.M, 2)
		# Allocate interval ranges
		grid_tensor = torch.zeros(self.M, n_points)
		# Fill in interval ranges
		for m, vals in enumerate(limits):
			grid_tensor[m, :] = torch.linspace(vals[0], vals[1], n_points)

		# Create meshgrid of points
		mesh = torch.meshgrid([grid for grid in grid_tensor])
		XX = torch.column_stack([x.ravel() for x in mesh])

		# Evaluate densities
		densities = self.density(
			torch.tensor(XX, device=self.device)).cpu().numpy()

		# Return ranges for M dimensions and all densities

		if self.M == 2:
			return (grid_tensor[0].numpy(), grid_tensor[1].numpy()), densities.reshape((n_points, n_points))
		else:
			return grid_tensor, densities

	def unit_test_multidimensional(self, limits, n_points=1000):
		grid_tensor, densities = self.eval_density_multi_grid(limits, n_points)
		total_density = densities.sum()
		if self.M == 2:
			scaling = (grid_tensor[0][-1]-grid_tensor[0][0])*(grid_tensor[1][-1]-grid_tensor[1][0])/(n_points**self.M)
		else:
			scaling = np.prod([(vals[-1]-vals[0]).item() for vals in grid_tensor])/(n_points**self.M)
		scaled_density = total_density * scaling
		return scaled_density

class TensorRingAlt(PyroModule):
	"""Tensor Ring model"""

	def __init__(self, Ks, device='cpu'):
		super().__init__()
		self.Ks = Ks
		self.device = device
		self.kwargs = {'Ks': Ks, 'device': device}
		self.M = len(self.Ks)
		self.train_losses = list()
		self.val_losses = list()

		# Weights for latent variable k_0
		self.k0_weights = PyroParam(
			torch.ones(self.Ks[0], device=self.device) / self.Ks[0],
			constraint=dist.constraints.simplex)

		# Parameters indexed by latent variable number
		# I.e. by 1, 2,..., M
		self.params = nn.ModuleDict()

		# Intialize weights, locs and scales
		self.init_params()

		if 'cuda' in self.device:
			self.cuda(self.device)
		
		if len(self.Ks) > 60:
			self.forward = self.forward_alt
		
	def init_params(self, loc_min=None, loc_max=None, scale_max=None):

		if loc_min is None:
			loc_min = [-1 for m in range(self.M)]
		if loc_max is None:
			loc_max = [1 for m in range(self.M)]
		if scale_max is None:
			scale_max = [1 for m in range(self.M)]

		# Initialize parameters for x_1, x_2,..., x_{M-1}
		for m in range(1, self.M):
			# PyroModule for storing weights, locs
			# and scales for x_m
			module = PyroModule(name=f'x_{m}')
			param_shape = (self.Ks[m-1], self.Ks[m])
			
			# Weight matrix W^m ("Transition probabilities")
			# I.e. probability of k_m = a given k_{m-1} = b
			module.weights = PyroParam(
				torch.ones(param_shape, device=self.device) / self.Ks[m],
				constraint=constraints.simplex)
			
			# Locs for x_m
			module.locs = PyroParam(
				dist.Uniform(loc_min[m-1]-1e-8, loc_max[m-1]+1e-8).sample(
					param_shape).to(self.device),
				constraint=constraints.real)
			
			# Scales for x_m
			module.scales = PyroParam(
				dist.Uniform(1e-8, scale_max[m-1]+1e-7).sample(
					param_shape).to(self.device),
				constraint=constraints.positive)
			
			self.params[str(m)] = module
		
		# Initialize parameters for x_M
		module = PyroModule(name=f'x_{self.M}')
		param_shape = (self.Ks[0], self.Ks[-1])

		# Locs for x_M
		module.locs = PyroParam(
			dist.Uniform(loc_min[m-1]-1e-8, loc_max[m-1]+1e-8).sample(
				param_shape).to(self.device),
			constraint=constraints.real)
			
		# Scales for x_M
		module.scales = PyroParam(
			dist.Uniform(1e-8, scale_max[m-1]+1e-7).sample(
				param_shape).to(self.device),
			constraint=constraints.positive)
			
		self.params[str(self.M)] = module

	
	@config_enumerate
	def forward(self, data=None, n_samples=1000):
		N, M = data.shape if data is not None else (n_samples, self.M)

		# Empty tensor for data samples
		x_sample = torch.empty(N, M, device=self.device)

		with pyro.plate('data', size=N):
			# Sample k_0
			k_0 = pyro.sample(
				'k_0',
				dist.Categorical(self.k0_weights))
			k_m_prev = k_0

			for m, params in enumerate(self.params.values()):
				# Sample k_m
				if m < self.M - 1:
					k_m = pyro.sample(
						f'k_{m + 1}',
						dist.Categorical(Vindex(params.weights)[k_m_prev]))
				else:
					k_m = k_m_prev
					k_m_prev = k_0

				# Observations of x_m
				obs = data[:, m] if data is not None else None

				# Sample x_m
				x_sample[:, m] = pyro.sample(
					f'x_{m + 1}',
					dist.Normal(loc=Vindex(params.locs)[k_m_prev, k_m],
								scale=Vindex(params.scales)[k_m_prev, k_m]+1e-9),
								obs=obs)

				k_m_prev = k_m

			return x_sample

	def guide(self, data):
		pass

	def fit_model(self, data, data_val=None, lr=3e-4, mb_size=512,
		n_epochs=500, verbose=True, early_stopping=False):
		N_train = len(data)
		if data_val is not None:
			N_val = len(data_val)
		else:
			early_stopping = False
		
		# Variables for early-stopping
		best_loss = np.inf
		count = 0

		adam = pyro.optim.Adam({"lr": lr})
		svi = SVI(self, self.guide, adam, loss=TraceEnum_ELBO())

		for epoch in range(n_epochs):
			self.train()

			mbs = BatchSampler(
				RandomSampler(range(N_train)),
				batch_size=mb_size,
				drop_last=False)

			loss = 0
			for mb_idx in mbs:
				loss += svi.step(data[mb_idx])
			
			self.train_losses.append(loss/N_train)
			
			if data_val is not None:
				self.eval()
				val_loss = self.nllh(data_val) / N_val
				self.val_losses.append(val_loss)

			if epoch % 10 == 0 and verbose:
				print('[epoch {}]  loss: {:.4f}'.format(epoch, loss/N_train))

			if early_stopping:
				# Reset counter if val loss has improved by 0.1%
				if val_loss < best_loss*(1 - 1e-3):
					best_loss = val_loss
					count = 0
				else:
					count += 1
				
				# Break training loop if val loss has not improved in 10 epochs
				if count == 10:
					break
				
	def hot_start(self, data, subsample_size=None, n_starts=100):
		seeds = torch.multinomial(
			torch.ones(10000) / 10000, num_samples=n_starts)
		inits = list()

		data_min = data.min(dim=0).values
		data_max = data.max(dim=0).values
		data_std = data.std(dim=0)

		if subsample_size is not None:
			subsample_idx = torch.randperm(len(data))[:subsample_size]
			data = data[subsample_idx]
	
		for seed in seeds:
			pyro.set_rng_seed(seed)
			pyro.clear_param_store()

			# Set new initial parameters
			self.init_params(loc_min=data_min,
							 loc_max=data_max,
							 scale_max=data_std)

			# Get initial loss
			self.fit_model(
				data, lr=0, mb_size=len(data), n_epochs=1, verbose=False)
			loss = self.train_losses[-1]

			# Save loss and seed
			inits.append((loss, seed))

			# Reset train losses
			self.train_losses = list()

		# Best initialization
		_, best_seed = min(inits)

		# Initialize with best seed
		pyro.set_rng_seed(best_seed)
		pyro.clear_param_store()
		self.init_params(loc_min=data_min,
						 loc_max=data_max,
						 scale_max=data_std)

	def log_density(self, data):
		with torch.no_grad():
			N, M = data.shape
			data = data.reshape(N, M, 1, 1) # N x M x 1 x 1
			
			# Initialize log density
			locs = self.params[str(self.M)].locs
			scales = self.params[str(self.M)].scales
			log_density = dist.Normal(locs, scales).log_prob(data[:, M-1]) # N x K_0 x K_{M-1}
			log_density = log_density.unsqueeze(2) # N x K_0 x 1 x K_{M-1}

			for m in reversed(range(self.M-1)):
				params = self.params[str(m+1)]

				log_prob = dist.Normal(
					params.locs, params.scales).log_prob(data[:, m]) # N x K_{m-1} x K_m
				log_weights = torch.log(params.weights) # 1 x K_{m-1} x K_m
				weighted_log_prob = (log_weights + log_prob).unsqueeze(1) # N x 1 x K_{m-1} x K_m
				
				if m == 0:
					weighted_log_prob = weighted_log_prob.permute(0, 2, 1, 3) # N x K_{m-1} x 1 x K_m
				
				log_density = torch.logsumexp(
					weighted_log_prob + log_density, dim=-1).unsqueeze(2) # N x K_0 x 1 x K_{m-1}
				
			log_density = log_density.squeeze(-1).squeeze(-1) # N x K_0
			log_k0_weights = torch.log(self.k0_weights) # K_0
			
			return torch.logsumexp(log_k0_weights + log_density, dim=-1) # N
	
	def nllh(self, data):
		with torch.no_grad():
			log_density = self.log_density(data)
			return -torch.sum(log_density).item()
	
	def eval_density_grid(self, n_points=100, grid=[-5, 5, -5, 5]):
		x_range = np.linspace(grid[0], grid[1], n_points)
		y_range = np.linspace(grid[2], grid[3], n_points)
		X1, X2 = np.meshgrid(x_range, y_range)
		XX = np.column_stack((X1.ravel(), X2.ravel()))
		densities = self.log_density(
			torch.tensor(XX, device=self.device)).cpu()
		densities = torch.exp(densities).numpy()

		return (x_range, y_range), densities.reshape((n_points, n_points))
	
	def unit_test_alt(self, n_points=100, grid=[-5, 5, -5, 5]):
		(x_range, y_range), density = self.eval_density_grid(n_points=n_points,
															 grid=grid)
		density = density.sum()
		height = (y_range[-1] - y_range[0])/len(y_range)
		width  = (x_range[-1] - x_range[0])/len(x_range)
		scaled_density = density*width*height

		return scaled_density