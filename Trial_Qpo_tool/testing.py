from prior import get_prior

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import tinygp
from tinygp import GaussianProcess, kernels
from stingray import Lightcurve
jax.config.update("jax_enable_x64", True)
import functools

# MAKING THE DATA
Times = np.linspace(0,1,256)

kernel = kernel = kernels.quasisep.Exp(
                scale = 1/jnp.exp(1.0), sigma = (jnp.exp(1.0))**0.5) + kernels.quasisep.Celerite(
                a = jnp.exp(-0.4), b = 0.0, c = jnp.exp(1), d = 2*jnp.pi*20)

def gaussian(t, mean_params):
    return mean_params["A"] * jnp.exp(-((t - mean_params["t0"])**2)/(2*(mean_params["sig"]**2)))

mean = functools.partial(gaussian, mean_params = {"A" : 3,    "t0" : 0.5,    "sig" : 0.2})

hqpogp = tinygp.GaussianProcess( kernel, Times, mean=mean)

counts = hqpogp.sample(jax.random.PRNGKey(101))
lightcurve = Lightcurve(Times, counts)

# MAKING THE GP

parameters = {
                # "kernel_type" : "qpo_plus_red_noise",
                # "mean_type" : "gaussian",
                "Times" : Times,
                "counts" : counts,
                "diag" : 0.1,
                "arn" : jnp.exp(1.0),    "crn" : jnp.exp(1.0),
                "aqpo": jnp.exp(-0.4),    "cqpo": jnp.exp(1),    "freq": 20,
                "A" : 3,    "t0" : 0.5,    "sig" : 0.2,
}

from GP import GP

model_type = ("QPO_plus_RN", "gaussian")

gp = GP(Lc = lightcurve, Model_type = model_type, Model_params = parameters)

# Small test

print(gp.maingp)
print(gp.maingp.kernel)
print(len(gp.maingp.X))
print(type(gp.maingp.X))

# Making the GPResult

from GP import GPResult

gpresult = GPResult(gp, prior_type = ("QPO_plus_RN", "gaussian"), prior_parameters=parameters)

print(len(gpresult.lc.time))
print(len(gpresult.lc.counts))

gpresult.run_sampling()