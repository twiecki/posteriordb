def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Parameters
        beta = pm.Flat("beta", shape=2)  # No prior specified in Stan
        sigma = pm.HalfFlat("sigma")     # real<lower=0> with no prior
        
        # Model
        mu = beta[0] + beta[1] * data['encouraged']
        watched_obs = pm.Normal("watched", mu=mu, sigma=sigma, observed=data['watched'])

    return model