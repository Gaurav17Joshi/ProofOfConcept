import tensorflow_probability.substrates.jax as tfp
from jax import random

from jaxns import ExactNestedSampler
from jaxns import Model
from jaxns import Prior
from jaxns import TerminationCondition

tfpd = tfp.distributions

def log_likelihood(theta):
    return 0.

def prior_model():
    x = yield Prior(tfpd.Uniform(0., 1.), name='x')
    return x

model = Model(prior_model=prior_model,
              log_likelihood=log_likelihood)

log_Z_true = 0.
print(f"True log(Z)={log_Z_true}")

# Create the nested sampler class. In this case without any tuning.
exact_ns = ExactNestedSampler(model=model, num_live_points=200, max_samples=1e4)

termination_reason, state = exact_ns(random.PRNGKey(42),
                                     term_cond=TerminationCondition(live_evidence_frac=1e-4))
results = exact_ns.to_results(state, termination_reason)
print("Completed")