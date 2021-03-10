# Imports
import pyro
import pyro.distributions as dist
import torch
from scipy.integrate import nquad
from pyro.infer import config_enumerate
from pyro.nn import PyroParam, PyroModule
from pyro.distributions import constraints
from sklearn.cluster import KMeans
from pyro.infer import TraceEnum_ELBO
from pyro.ops.indexing import Vindex


class gmm(PyroModule):
    """
    Gaussian Mixture Model (GMM) for density estimation.
    """

    def __init__(self, K, M=2, N=1000):
        super(gmm, self).__init__()
        self.K = K
        self.M = M
        self.N = N
        self.initialized = False

        # Initialise GMM parameters
        self.loc = torch.normal(torch.zeros((self.K, self.M)), torch.ones((self.K, self.M)))
        self.scale = torch.ones((self.K, self.M))
        self.weight = torch.ones(self.K)

        # self.loc = PyroParam(torch.randn(self.K, self.M))
        # self.scale = PyroParam(torch.ones(self.K, self.M), constraint=constraints.positive)
        # self.weight = PyroParam(torch.ones(self.K)/self.K, constraint=constraints.simplex)

    @config_enumerate
    def model(self, data=None):

        if not self.initialized and data is not None:
            self.initialize_param(data)
            self.N = len(data)

        # Declare parameters with constraints
        loc = pyro.param("loc", self.loc)
        scale = pyro.param("scale", self.scale, constraints.positive)
        weight = pyro.param("weight", self.weight, constraints.simplex)

        with pyro.plate("data", self.N):
            # Draw assignment

            assignment = pyro.sample('assignment', dist.Categorical(weight)).long()
            #Unclear if necessary: fn_dist = dist.Normal(Vindex(loc)[..., assignment, :], Vindex(scale)[..., assignment, :]).to_event(1)

            # Define distribution and sample
            fn_dist = dist.Normal(loc[assignment, :], scale[assignment, :]).to_event(1)
            samples = pyro.sample("samples", fn_dist, obs=data)

        # Save parameters to model
        self.loc = loc
        self.scale = scale
        self.weight = weight
        return samples

    def guide(self, data):
        """
        Empty guide for MLE
        """
        pass

    def get_likelihood(self, data):
        """
        Function to compute likelihood manually.
        """

        likelihood = torch.zeros(len(data))

        with torch.no_grad():
            fn_dist = dist.Normal(self.loc, self.scale).to_event(1)
            for i in range(len(data)):
                tmp = torch.log((self.weight * torch.exp(fn_dist.log_prob(data[i]))).sum())
                likelihood[i] = tmp if not torch.isinf(tmp) else -100

            return likelihood.numpy()

    def get_likelihood_alt(self, data, num_particles=1):
        """
        Function to compute likelihood manually.
        """
        trace_elbo = TraceEnum_ELBO(num_particles=num_particles)
        for model_trace, _ in trace_elbo._get_traces(self.model, self.guide, [data], {}):
            grid_log_probs = model_trace.nodes["samples"]["log_prob"]
        #grid_log_probs = grid_log_probs.reshape(data.shape[0], data.shape[1])
        return grid_log_probs


    def initialize_param(self, data):
        km = KMeans(n_clusters=self.K, n_init=5)
        loc = km.fit(data.numpy()).cluster_centers_
        self.loc = torch.from_numpy(loc)
        self.initialized = True

    def get_density(self, data):
        # Computes the total density of the input data
        ll = self.get_likelihood_alt(data)
        density = torch.exp(ll).sum()
        return density

    def unit_test(self, int_limits):
        """Integrates probablity density"""
        return nquad(
            lambda *args: self.get_density(torch.tensor([args])).item(),
            int_limits)


class CP(PyroModule):
    def __init__(self, K, M=2, N=1000):
        super(CP, self).__init__()
        self.K = K
        self.M = M
        self.N = N
        self.initialized = False

        # Initialise parameters
        self.loc = PyroParam(torch.randn(self.K, self.M))
        self.scale = PyroParam(torch.ones(self.K, self.M), constraint=constraints.positive)
        self.weight = PyroParam(torch.ones(self.K)/self.K, constraint=constraints.simplex)


    @config_enumerate
    def forward(self, data=None):

        if not self.initialized:
            self.initialize_param(data)
        if data is not None:
            self.N = len(data)

        # Allocate samples
        samples = torch.zeros((self.N, self.M))

        with pyro.plate("data", self.N):
            # Draw assignment
            assignment = pyro.sample('assignment', dist.Categorical(self.weight)).long()
            for m in range(self.M):
                obs = data[:, m] if data is not None else None
                samples[:, m] = pyro.sample(f'samples_{m}', dist.Normal(loc=self.loc[assignment, m],
                                                                        scale=self.scale[assignment, m]),
                                                                        obs=obs)

        return samples

    def model(self, data=None):
        return self.forward(data)

    def guide(self, data):
        """
        Empty guide for MLE
        """
        pass

    def get_likelihood(self, data):
        """
        Function to compute likelihood manually.
        """

        likelihood = torch.zeros(len(data))

        with torch.no_grad():
            fn_dist = dist.Normal(self.loc, self.scale).to_event(1)
            for i in range(len(data)):
                tmp = torch.log((self.weight * torch.exp(fn_dist.log_prob(data[i]))).sum())
                likelihood[i] = tmp if not torch.isinf(tmp) else -100

            return likelihood.numpy()

    def get_likelihood_alt(self, data, num_particles=1):
        """
        Function to compute log-likelihood via TraceEnum_ELBO
        Messy.
        """
        with torch.no_grad():
            # Extracts logprob for all datapoints in all clusters
            trace_elbo = TraceEnum_ELBO(num_particles=num_particles)
            grid_log_probs = torch.zeros((self.K, data.shape[0]))  # Shape [8, 5000] for 5000 datapoints and 8 clusters
            for model_trace, _ in trace_elbo._get_traces(self.model, self.guide, [data], {}):
                # Sum log-likelihoods over dimensions
                for m in range(self.M):
                    grid_log_probs += model_trace.nodes[f"samples_{m}"]["log_prob"]

            # Scales the log-likelihood values by the weights
            ll = torch.zeros(data.shape[0])
            for i in range(data.shape[0]):
                for j in range(self.K):
                    ll[i] += grid_log_probs[j, i].exp()*self.weight[j]
                ll[i] = torch.log(ll[i])

        return ll

    def get_density(self, data):
        # Computes the total density of the input data
        ll = self.get_likelihood_alt(data)
        density = torch.exp(ll).sum()
        return density

    def initialize_param(self, data):
        km = KMeans(n_clusters=self.K, n_init=5)
        loc = km.fit(data.numpy()).cluster_centers_
        self.loc = torch.from_numpy(loc)
        self.initialized = True

    def unit_test(self, int_limits):
        """Integrates probablity density"""
        return nquad(
            lambda *args: self.get_density(torch.tensor([args])).item(),
            int_limits)

