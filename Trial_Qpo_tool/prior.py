import jax.numpy as jnp
import tensorflow_probability as tfp
from jaxns import Prior

tfpd = tfp.distributions
tfpb = tfp.bijectors

def get_prior(**kwargs):
    """
    A prior generator function based on given values

    Parameters
    ----------
    kwargs:
        All possible keyword arguments to construct the prior.

    Returns
    -------
    The Prior function.
    The arguments of the prior function are in the order of
    Kernel arguments (RN arguments, QPO arguments), 
    Mean arguments
    Non Windowed arguments

    """
    kwargs["T"] = kwargs["Times"][-1] - kwargs["Times"][0]    # Total time
    kwargs["f"] = 1/(kwargs["Times"][1] - kwargs["Times"][0]) # Sampling frequency
    kwargs["min"] = jnp.min(kwargs["lightcurve"])
    kwargs["max"] = jnp.max(kwargs["lightcurve"])
    kwargs["span"] = kwargs["max"] - kwargs["min"]

    kernel_prior_args = get_kernel_prior_args(**kwargs)
    mean_prior_args = get_mean_prior_args(**kwargs)
    # non_windowed_prior_args = get_non_windowed_prior_args(**kwargs)

    def prior_model():
        return kernel_prior_args , mean_prior_args #, non_windowed_prior_args
    
    return prior_model


def get_kernel_prior_args(kernel_type, **kwargs):

    priors = kernel_prior_getters[kernel_type](**kwargs)
    return priors


def get_mean_prior_args(mean_type, **kwargs):

    priors = mean_prior_getters[mean_type](**kwargs)
    return priors

def get_non_windowed_prior_args():

    pass

def _get_red_noise_prior(**kwargs):

    arn = yield Prior(tfpd.Uniform(0.1*kwargs["span"], 2*kwargs["span"]), name='arn') 
    crn = yield Prior(tfpd.Uniform(jnp.log(1/kwargs["T"]), jnp.log(kwargs["f"])), name='crn')

    return arn, crn


def _get_qpo_plus_red_noise_prior(**kwargs):

    arn = yield Prior(tfpd.Uniform(0.1*kwargs["span"], 2*kwargs["span"]), name='arn')
    crn = yield Prior(tfpd.Uniform(jnp.log(1/kwargs["T"]), jnp.log(kwargs["f"])), name='crn')

    aqpo = yield Prior(tfpd.Uniform(0.1*kwargs["span"], 2*kwargs["span"]), name='aqpo')
    cqpo = yield Prior(tfpd.Uniform(1/10/kwargs["T"], jnp.log(kwargs["f"])), name='cqpo')
    freq = yield Prior(tfpd.Uniform(2/kwargs["T"], kwargs["f"]/2 ), name='freq')
    
    return arn, crn, aqpo, cqpo, freq

def _get_constant_mean_prior(**kwargs):

    pass

def _get_gaussian_mean_prior(**kwargs):
        
    A = yield Prior(tfpd.Uniform(0.1*kwargs["span"], 2*kwargs["span"]), name='A')
    t0 = yield Prior(tfpd.Uniform(kwargs["Times"][0]-0.1*kwargs["T"], 
                                  kwargs["Times"][-1]+0.1*kwargs["T"]), name='t0')
    sig = yield Prior(tfpd.Uniform(0.5*1/kwargs["f"], 2*kwargs["T"]), name='sig')
    
    return A, t0, sig

def _get_exponential_mean_prior(**kwargs):
    A = yield Prior(tfpd.Uniform(0.1*kwargs["span"], 2*kwargs["span"]), name='A')
    t0 = yield Prior(tfpd.Uniform(kwargs["Times"][0]-0.1*kwargs["T"], 
                                  kwargs["Times"][-1]+0.1*kwargs["T"]), name='t0')
    sig = yield Prior(tfpd.Uniform(0.5*1/kwargs["f"], 2*kwargs["T"]), name='sig')
    
    return A, t0, sig

kernel_prior_getters = dict(
    red_noise=_get_red_noise_prior, 
    qpo_plus_red_noise=_get_qpo_plus_red_noise_prior
)

mean_prior_getters = dict(
    constant=_get_constant_mean_prior,
    gaussian = _get_gaussian_mean_prior,
    exponential = _get_exponential_mean_prior
)