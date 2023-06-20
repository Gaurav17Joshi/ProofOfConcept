import numpy as np
import jax.numpy as jnp
from jaxns.special_priors import Poisson, Beta, ForcedIdentifiability
from jaxns.prior import Prior
from jax import random

def test_poisson():
    key = random.PRNGKey(0)
    mu = 2.5
    poisson_prior = Poisson("poisson_prior", mu)

    for _ in range(100):
        key, subkey = random.split(key)
        sample = poisson_prior.sample(subkey)
        assert jnp.all(sample >= 0)
        assert jnp.all(Prior.log_prior_fn(poisson_prior)(sample) <= 0.)

def test_beta():
    key = random.PRNGKey(0)
    alpha = 1.0
    beta = 2.0
    beta_prior = Beta("beta_prior", alpha, beta)

    for _ in range(100):
        key, subkey = random.split(key)
        sample = beta_prior.sample(subkey)
        assert jnp.all((sample >= 0.) & (sample <= 1.))
        assert jnp.all(Prior.log_prior_fn(beta_prior)(sample) <= 0.)

def test_forced_identifiability():
    key = random.PRNGKey(0)
    num_samples = 10
    forced_identifiability_prior = ForcedIdentifiability("forced_identifiability_prior", num_samples)

    for _ in range(100):
        key, subkey = random.split(key)
        samples = forced_identifiability_prior.sample(subkey)
        assert jnp.all((samples >= 0.) & (samples <= 1.))
        assert jnp.all(jnp.diff(samples) >= 0.)
        assert jnp.all(Prior.log_prior_fn(forced_identifiability_prior)(samples) <= 0.)

if __name__ == '__main__':
    test_poisson()
    test_beta()
    test_forced_identifiability()
    print("All tests passed.")