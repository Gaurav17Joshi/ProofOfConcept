import jax.numpy as jnp

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

    kernel_prior_args = get_kernel_prior_args(**kwargs)
    mean_prior_args = get_mean_prior_args(**kwargs)
    non_windowed_prior_args = get_non_windowed_prior_args(**kwargs)

    def prior_model():
        return kernel_prior_args , mean_prior_args , non_windowed_prior_args
    
    return prior_model


def get_kernel_prior_args():
    
    pass

def get_mean_prior_args():

    pass

def get_non_windowed_prior_args():

    pass

