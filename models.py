# Imports
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import config_enumerate
from torch import nn
from torch.distributions import constraints


class gmm(nn.Module):
    """
    Gaussian Mixture Model (GMM) for density estimation.
    """
    def __init__(self, K):
        super(gmm, self).__init__()
        self.K = K

        # Initialise GMM parameters. Mean values are drawn from standard normal distribution.
        self.loc = torch.normal(torch.zeros((self.K, 2)), torch.ones((self.K, 2)))
        self.scale = torch.ones((self.K, 2))
        self.weight = torch.ones(self.K)

    @config_enumerate
    def model(self, data=None):
        N = len(data) if data is not None else 1000

        # Declare parameters with constraints
        loc = pyro.param("loc", self.loc)
        scale = pyro.param("scale", self.scale, constraints.positive)
        weight = pyro.param("weight", self.weight, constraints.simplex)

        with pyro.plate("data", N):
            # Draw assignment
            assignment = pyro.sample('assignment', dist.Categorical(weight)).long()  # Unclear if necessary: fn_dist = dist.Normal(Vindex(loc)[..., assignment, :], Vindex(scale)[..., assignment, :]).to_event(1)

            # Define distribution and sample
            fn_dist = dist.Normal(loc[assignment, :], scale[assignment, :]).to_event(1)
            x_samples = pyro.sample("x_samples", fn_dist, obs=data)

        # Save parameters to model
        self.loc = loc
        self.scale = scale
        self.weight = weight
        return x_samples

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
                #tmp = (self.weight*torch.exp(fn_dist.log_prob(data[i]))).sum()
                tmp = torch.log((self.weight*torch.exp(fn_dist.log_prob(data[i]))).sum())
                likelihood[i] = tmp if not torch.isinf(tmp) else -100

            return likelihood.numpy()

