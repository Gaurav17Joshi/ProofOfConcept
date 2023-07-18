# Dependencies
import numpy as np
import matplotlib.pyplot as plt

import jax
from jax import random
import jax.numpy as jnp
# jax.config.update("jax_enable_x64", True)

import tinygp
import functools
from jaxns import ExactNestedSampler, TerminationCondition, Prior, Model

import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions
tfpb = tfp.bijectors

# Data -----------------------------------------------------------------------
# Times = np.linspace(0,1,64)
Times = jnp.linspace(0,1,64) # works well, using np.linpace requires us to use numpy.float64
kernel = tinygp.kernels.quasisep.Exp(scale=1 / jnp.exp(1.0), sigma=(jnp.exp(1.5)) ** 0.5)

# Remove mean in another file
def _gaussian(t, mean_params):
    A = jnp.atleast_1d(mean_params["A"])[:, jnp.newaxis]
    t0 = jnp.atleast_1d(mean_params["t0"])[:, jnp.newaxis]
    sig = jnp.atleast_1d(mean_params["sig"])[:, jnp.newaxis]
    return jnp.sum(A * jnp.exp(-((t - t0) ** 2) / (2 * (sig**2))), axis=0)

mean_params = {"A" : jnp.array([3.0]), "t0" : jnp.array([0.2]), "sig" : jnp.array([0.2]) }
mean = functools.partial(_gaussian, mean_params=mean_params)

gp = tinygp.GaussianProcess(kernel = kernel, X = Times, mean_value = mean(Times))
counts =  gp.sample(key = jax.random.PRNGKey(6))

# Priors and Likelihood ------------------------------------------------------

T = Times[-1] - Times[0]
f = 1/(Times[1]- Times[0])
span = jnp.max(counts) - jnp.min(counts)

# Here, we have made mutiple mean function with 2 gaussians.
def prior_model():
    arn = yield Prior(tfpd.Uniform(low = 0.1 * span, high = 2 * span), name='arn')
    crn = yield Prior(tfpd.Uniform(low = jnp.log(1 / T), high = jnp.log(f)), name='crn')
    A = yield Prior(tfpd.Uniform(low = 0.1 * span, high = 2 * span), name='A')
    t0 = yield Prior(tfpd.Uniform(low = Times[0] - 0.1*T, high = Times[-1] + 0.1*T), name = 't0')
    sig = yield Prior(tfpd.Uniform(low = 0.5 * 1 / f, high = 2 * T), name='sig')

    return arn, crn, A, t0, sig

def likelihood_model(arn,crn, A, t0, sig):
    kernel = tinygp.kernels.quasisep.Exp(scale=1 /crn, sigma=(arn) ** 0.5)
    mean_params = {"A": A, "t0": t0, "sig": sig}
    mean = functools.partial(_gaussian, mean_params=mean_params)
    gp = tinygp.GaussianProcess(kernel, Times, mean_value=mean(Times))
    return gp.log_probability(counts)

# Nested Sampling ------------------------------------------------------------
NSmodel = Model(prior_model=prior_model, log_likelihood=likelihood_model)
NSmodel.sanity_check(random.PRNGKey(10), S=100)

Exact_ns = ExactNestedSampler(NSmodel, num_live_points=500, max_samples=1e4)
Termination_reason, State = Exact_ns(
            random.PRNGKey(42), term_cond=TerminationCondition(live_evidence_frac=1e-3))
Results = Exact_ns.to_results(State, Termination_reason)
print("Simulation Complete")