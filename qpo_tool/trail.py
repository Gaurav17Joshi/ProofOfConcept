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

Times = np.linspace(0,1,256)

def gaussian(t, mean_params):
    return mean_params["A"] * jnp.exp(-((t - mean_params["t0"])**2)/(2*(mean_params["sig"]**2)))

# Build Gp
def build_gp(kernel_params, mean_params, t, kernel_type, mean_type = "gaussian"):
    if kernel_type == "QPO_plus_RN":
        kernel = kernels.quasisep.Exp(
                scale = 1/kernel_params["crn"], sigma = (kernel_params["arn"])**0.5) + kernels.quasisep.Celerite(
                a = kernel_params["aqpo"], b = 0.0, c = kernel_params["cqpo"], d = 2*jnp.pi*kernel_params["freq"])
    elif kernel_type == "RN":
        kernel = kernels.quasisep.Exp(
                scale = 1/kernel_params["crn"], sigma = (kernel_params["arn"])**0.5)
    
    # Using partial to make the mean so that it only takes the time value
    if mean_type == "gaussian":
        mean = functools.partial(gaussian, mean_params = mean_params)
        return tinygp.GaussianProcess( kernel, t, mean=mean)
    elif mean_type == "constant":
        mean = mean_params
        return tinygp.GaussianProcess( kernel, t)

hqpoparams = {
    "arn" : jnp.exp(1.0),    "crn" : jnp.exp(1.0),
    "aqpo": jnp.exp(-0.4),    "cqpo": jnp.exp(1),    "freq": 20,
}

hqpokernel = kernels.quasisep.Exp(
    scale = 1/hqpoparams["crn"], sigma = (hqpoparams["arn"])**0.5) + kernels.quasisep.Celerite(
        a = hqpoparams["aqpo"], b = 0.0, c = hqpoparams["cqpo"], d = 2*jnp.pi*hqpoparams["freq"])

mean_params = {
    "A" : 3,    "t0" : 0.5,    "sig" : 0.2,
}

hqpogp = build_gp(hqpoparams, mean_params, Times, kernel_type = "QPO_plus_RN")

lightcurve = hqpogp.sample(jax.random.PRNGKey(4))

"""

TRYING TO GET THE PRIOR PARAMETERS

"""

parameters = {"kernel_type" : "qpo_plus_red_noise",
                "mean_type" : "gaussian",
                "Times" : Times,
                "lightcurve" : lightcurve,
}

prior_model = get_prior(**parameters)
prior_objects = prior_model()

for prior_obj in prior_objects:
    # for objs in prior_obj:
    #     print(objs)
    print(prior_obj)